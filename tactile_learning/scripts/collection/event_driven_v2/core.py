from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
import math


@dataclass(frozen=True)
class Config:
    capture_hz: float = 50.0
    precontact_frames: int = 4
    hold_frames: int = 8
    rate_0: float = 0.35
    rate_1: float = 0.12
    rate_2: float = 0.06
    rate_3: float = 0.0
    stable_time_s: float = 0.04
    max_wait_first_s: float = 6.0
    max_wait_three_s: float = 6.0
    close_limit_grace_s: float = 0.20

    def __post_init__(self):
        for k in ("capture_hz", "stable_time_s", "max_wait_first_s", "max_wait_three_s"):
            v = getattr(self, k)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{k} must be finite and positive")
        if self.precontact_frames < 0 or self.hold_frames < 0:
            raise ValueError("frame counts must be nonnegative")
        rates = (self.rate_0, self.rate_1, self.rate_2, self.rate_3)
        if any(not math.isfinite(x) or x < 0 for x in rates):
            raise ValueError("rates must be finite and nonnegative")
        if not (self.rate_0 >= self.rate_1 >= self.rate_2 >= self.rate_3):
            raise ValueError("rates must decrease with contact count")

    def as_dict(self):
        return asdict(self)


class State(str, Enum):
    WAIT_FIRST = "wait_first_contact"
    DYNAMIC = "dynamic_contact"
    CONFIRM = "confirm_three_contact"
    HOLD = "hold"
    PARTIAL = "partial_contact"
    DONE = "done"
    FAILED = "failed"


_ALLOWED = {
    State.WAIT_FIRST: {State.DYNAMIC, State.FAILED},
    State.DYNAMIC: {State.CONFIRM, State.PARTIAL, State.FAILED},
    State.CONFIRM: {State.DYNAMIC, State.HOLD, State.PARTIAL, State.FAILED},
    State.HOLD: {State.DONE, State.PARTIAL, State.FAILED},
    State.PARTIAL: set(), State.DONE: set(), State.FAILED: set(),
}


class StateMachine:
    def __init__(self):
        self.state = State.WAIT_FIRST
        self.history = [self.state.value]

    def set(self, target):
        if target == self.state:
            return
        if target not in _ALLOWED[self.state]:
            raise RuntimeError(f"illegal transition {self.state.value} -> {target.value}")
        self.state = target
        self.history.append(target.value)


class GripperController:
    """Rates are scalar tendon-command-coordinate rates, not measured hinge speeds."""
    def __init__(self, close_sign, close_limit, rates=(0.35, 0.12, 0.06, 0.0)):
        if close_sign not in (-1, 1):
            raise ValueError("close_sign must be -1 or +1")
        if not (rates[0] >= rates[1] >= rates[2] >= rates[3] >= 0):
            raise ValueError("invalid rate schedule")
        self.sign, self.limit, self.rates = int(close_sign), float(close_limit), tuple(rates)

    def at_limit(self, u):
        return u <= self.limit + 1e-12 if self.sign < 0 else u >= self.limit - 1e-12

    def step(self, u, dt, contacts):
        if dt <= 0 or contacts not in (0, 1, 2, 3):
            raise ValueError("invalid dt/contact count")
        if contacts == 3 or self.at_limit(u):
            return float(u), 0.0
        rate = self.rates[contacts]
        nxt = u + self.sign * rate * dt
        nxt = max(nxt, self.limit) if self.sign < 0 else min(nxt, self.limit)
        return float(nxt), float(rate)


class ContactTracker:
    def __init__(self, dt, stable_time_s):
        self.required = max(1, int(math.ceil(stable_time_s / dt - 1e-12)))
        self.first = [None, None, None]
        self.first_time = None
        self.first_finger = None
        self.three_time = None
        self.stable_time = None
        self.stable_steps = 0
        self.hold_lost = False

    def update(self, contact_states, sim_time):
        bits = tuple(bool(x["contact"]) for x in contact_states)
        names = ("finger1", "finger2", "finger3")
        for i, on in enumerate(bits):
            if on and self.first[i] is None:
                self.first[i] = float(sim_time)
                if self.first_time is None:
                    self.first_time, self.first_finger = float(sim_time), names[i]
        all_on = all(bits)
        if all_on:
            if self.three_time is None:
                self.three_time = float(sim_time)
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        stable = self.stable_steps >= self.required
        if stable and self.stable_time is None:
            self.stable_time = float(sim_time)
        return {"bits": bits, "count": sum(bits), "any": any(bits),
                "all": all_on, "stable": stable}

    def as_dict(self):
        return {
            "first_contact_finger": self.first_finger,
            "first_contact_time_s": self.first_time,
            "finger_first_contact_times_s": {
                f"finger{i+1}": self.first[i] for i in range(3)
            },
            "three_contact_time_s": self.three_time,
            "stable_contact_time_s": self.stable_time,
            "stable_required_steps": self.required,
            "hold_contact_lost": self.hold_lost,
        }


class EventBuffer:
    def __init__(self, pre_frames):
        self.pre = deque(maxlen=pre_frames)
        self.samples = []
        self.started = False
        self.dynamic_frames = 0
        self.hold_frames = 0

    def add_pre(self, sample):
        self.pre.append(sample)

    def start_dynamic(self, sample):
        if self.started:
            raise RuntimeError("dynamic already started")
        self.samples.extend(list(self.pre))
        self.pre.clear()
        sample["phase"] = "dynamic_contact"
        self.samples.append(sample)
        self.started = True
        self.dynamic_frames = 1

    def add_dynamic(self, sample):
        sample["phase"] = "dynamic_contact"
        self.samples.append(sample)
        self.dynamic_frames += 1

    def add_hold(self, sample):
        sample["phase"] = "hold"
        self.samples.append(sample)
        self.hold_frames += 1

    def finish(self):
        if not self.started:
            self.samples.extend(list(self.pre))
            self.pre.clear()
        return self.samples


def assess(events, samples, required_hold):
    """Classify an episode without discarding useful partial-contact data.

    success: stable simultaneous three-finger contact plus hold.
    partial_contact: at least one finger contacted, but three-finger stability
                     was not achieved before the close/time limit.
    no_contact: no finger contacted; samples are retained for audit but are
                not a tactile-classification training example.
    """
    if not samples:
        return False, "empty_episode"
    times = [float(s["time"]) for s in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        return False, "non_monotonic_sample_time"

    if events.get("first_contact_time_s") is None:
        return False, "no_contact"

    if events.get("stable_contact_time_s") is None:
        return True, "partial_contact"

    if events.get("hold_contact_lost"):
        return True, "partial_contact_hold_lost"

    actual_hold = sum(s.get("phase") == "hold" for s in samples)
    if actual_hold < required_hold:
        return True, "partial_contact_insufficient_hold"

    return True, "success"
