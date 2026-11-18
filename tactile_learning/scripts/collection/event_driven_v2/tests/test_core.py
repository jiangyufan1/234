import unittest
from scripts.collection.event_driven_v2.core import (
    Config, ContactTracker, EventBuffer, GripperController,
    State, StateMachine, assess,
)

def cs(bits):
    return [{"contact": bool(x)} for x in bits]

class V2CoreTests(unittest.TestCase):
    def test_config(self):
        c = Config()
        self.assertEqual((c.precontact_frames, c.hold_frames), (4, 8))

    def test_rates(self):
        g = GripperController(-1, -1.2)
        self.assertEqual([g.rates[i] for i in range(4)], [0.35, 0.12, 0.06, 0.0])

    def test_three_contacts_freeze(self):
        g = GripperController(-1, -1.2)
        self.assertEqual(g.step(-0.8, 0.01, 3), (-0.8, 0.0))

    def test_negative_limit(self):
        g = GripperController(-1, -1.2)
        self.assertEqual(g.step(-1.199, 0.01, 0), (-1.2, 0.35))

    def test_positive_limit(self):
        g = GripperController(1, 1.2)
        self.assertEqual(g.step(1.199, 0.01, 0), (1.2, 0.35))

    def test_contact_events(self):
        t = ContactTracker(0.001, 0.04)
        t.update(cs([1,0,0]), 1.0)
        t.update(cs([1,1,0]), 1.2)
        t.update(cs([1,1,1]), 1.4)
        d = t.as_dict()
        self.assertEqual(d["first_contact_finger"], "finger1")
        self.assertEqual(d["finger_first_contact_times_s"]["finger2"], 1.2)
        self.assertEqual(d["three_contact_time_s"], 1.4)

    def test_stability_debounce(self):
        t = ContactTracker(0.01, 0.04)
        for i in range(3):
            r = t.update(cs([1,1,1]), i * 0.01)
        self.assertFalse(r["stable"])
        self.assertTrue(t.update(cs([1,1,1]), 0.03)["stable"])

    def test_break_resets_stability(self):
        t = ContactTracker(0.01, 0.04)
        t.update(cs([1,1,1]), 0.00)
        t.update(cs([1,1,1]), 0.01)
        t.update(cs([1,0,1]), 0.02)
        self.assertEqual(t.stable_steps, 0)

    def test_state_transitions(self):
        s = StateMachine()
        s.set(State.DYNAMIC)
        s.set(State.CONFIRM)
        s.set(State.HOLD)
        s.set(State.DONE)
        self.assertEqual(s.state, State.DONE)

    def test_illegal_state_transition(self):
        with self.assertRaises(RuntimeError):
            StateMachine().set(State.HOLD)

    def test_prebuffer(self):
        b = EventBuffer(4)
        for i in range(6):
            b.add_pre({"time": i, "phase": "precontact"})
        b.start_dynamic({"time": 6})
        self.assertEqual([x["time"] for x in b.samples], [2,3,4,5,6])

    def test_variable_dynamic_and_hold(self):
        b = EventBuffer(2)
        b.start_dynamic({"time": 1})
        b.add_dynamic({"time": 2})
        b.add_hold({"time": 3})
        self.assertEqual((b.dynamic_frames, b.hold_frames), (2,1))

    def test_quality_success(self):
        e = {"first_contact_time_s": 1.0, "stable_contact_time_s": 1.2,
             "hold_contact_lost": False}
        samples = [{"time":1.0,"phase":"dynamic_contact"},
                   {"time":1.1,"phase":"hold"}]
        self.assertEqual(assess(e, samples, 1), (True, "success"))

    def test_partial_contact_is_retained_as_valid_data(self):
        events = {
            "first_contact_time_s": 1.0,
            "stable_contact_time_s": None,
            "hold_contact_lost": False,
        }
        samples = [
            {"time": 1.0, "phase": "dynamic_contact"},
            {"time": 1.02, "phase": "dynamic_contact"},
        ]
        self.assertEqual(assess(events, samples, 8), (True, "partial_contact"))

    def test_no_contact_is_not_training_eligible(self):
        events = {
            "first_contact_time_s": None,
            "stable_contact_time_s": None,
            "hold_contact_lost": False,
        }
        samples = [{"time": 0.0, "phase": "precontact"}]
        self.assertEqual(assess(events, samples, 8), (False, "no_contact"))

    def test_quality_rejects_hold_loss(self):
        e = {"first_contact_time_s": 1.0, "stable_contact_time_s": 1.2,
             "hold_contact_lost": True}
        self.assertEqual(assess(e, [{"time":1.0,"phase":"hold"}], 1), (True, "partial_contact_hold_lost"))

if __name__ == "__main__":
    unittest.main()
