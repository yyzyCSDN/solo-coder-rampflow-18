import unittest
from datetime import datetime, timezone

from rampflow.crew_duty_control import (
    CrewMember,
    CrewDutyController,
    DutySegment,
    DutyViolation,
)
from rampflow.models import Decision
from rampflow.resource_dispatch import (
    DispatchError,
    DispatchRequest,
    MobileResource,
    ResourceDispatcher,
    TravelGraph,
)
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


class CrewRestTests(unittest.TestCase):
    def member(self):
        return CrewMember(
            "C1",
            frozenset({"push"}),
            "A",
            max_duty_min=600,
            min_rest_min=60,
            max_continuous_work_min=100,
        )

    def test_short_break_does_not_reset_continuous_work(self):
        controller = CrewDutyController([self.member()])
        controller.assign("C1", DutySegment("s1", Window(D(10), D(11)), "A", "push"))
        # Only a 10-minute break: not a real rest, so the 50 minutes after it
        # accumulate onto the previous 60 and must breach the 100-minute cap.
        probe = DutySegment("s2", Window(D(11, 10), D(12)), "A", "push")
        ok, _ = controller.can_accept("C1", probe)
        self.assertFalse(ok)
        with self.assertRaises(DutyViolation):
            controller.assign("C1", probe)

    def test_real_rest_resets_continuous_work(self):
        controller = CrewDutyController([self.member()])
        controller.assign("C1", DutySegment("s1", Window(D(10), D(11)), "A", "push"))
        # 90 minutes off meets the 60-minute minimum rest: a genuine reset.
        ok, _ = controller.can_accept(
            "C1", DutySegment("s2", Window(D(12, 30), D(13, 20)), "A", "push")
        )
        self.assertTrue(ok)


class VehicleReadinessTests(unittest.TestCase):
    def dispatcher(self, available_at):
        graph = TravelGraph()
        graph.add("A", "B", 6000)  # 6000 m at 10 m/s = 10 minutes of travel
        resource = MobileResource("V1", "tug", frozenset(), "A", available_at, 10)
        return ResourceDispatcher([resource], graph)

    def test_plan_waits_for_vehicle_availability(self):
        dispatcher = self.dispatcher(D(10, 30))
        request = DispatchRequest("Q1", "tug", None, "B", D(10), 5, D(11))
        plan = dispatcher.plan(request)
        # The vehicle is busy until 10:30, so the plan must not pretend it
        # departed at 09:50 to meet the 10:00 request time.
        self.assertEqual(plan.depart_at, D(10, 30))
        self.assertEqual(plan.arrive_at, D(10, 40))
        self.assertEqual(plan.service, Window(D(10, 40), D(10, 45)))
        dispatcher.commit(plan)
        self.assertEqual(dispatcher.state("V1").available_at, D(10, 45))

    def test_plan_refuses_when_vehicle_cannot_make_it(self):
        dispatcher = self.dispatcher(D(10, 30))
        request = DispatchRequest("Q2", "tug", None, "B", D(10), 5, D(10, 44))
        with self.assertRaises(DispatchError):
            dispatcher.plan(request)
        # Refusal must be clean: nothing reserved, vehicle untouched.
        self.assertEqual(dispatcher.dispatches, {})
        self.assertEqual(dispatcher.state("V1").location, "A")
        self.assertEqual(dispatcher.state("V1").available_at, D(10, 30))


class MobileCrewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.at = D(10)

    def test_failed_assignment_publishes_nothing(self):
        control = RampWorkflowControl()

        def action():
            raise RuntimeError("crew over duty")

        with self.assertRaises(RuntimeError):
            control.assign_mobile_crew("op-1", "C1", self.at, action)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_rejected_requirement_blocks_assignment(self):
        control = RampWorkflowControl()
        calls = []
        denied = Decision(False, "rest", "minimum rest not met")
        with self.assertRaises(WorkflowRejected):
            control.assign_mobile_crew(
                "op-2", "C1", self.at,
                lambda: calls.append("commit"),
                requirements=(denied,),
            )
        self.assertEqual(calls, [])
        self.assertEqual(control.stream.head, 0)

    def test_assignment_replay_is_idempotent(self):
        control = RampWorkflowControl()
        calls = []
        first = control.assign_mobile_crew(
            "op-3", "C1", self.at, lambda: calls.append("commit") or "ok"
        )
        second = control.assign_mobile_crew(
            "op-3", "C1", self.at, lambda: calls.append("duplicate")
        )
        self.assertIs(first, second)
        self.assertEqual(calls, ["commit"])
        self.assertEqual(control.stream.head, 1)


if __name__ == "__main__":
    unittest.main()
