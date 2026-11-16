"""Check inter-finger sensor clearance independently of MuJoCo collision masks.

The gels and their housings deliberately overlap within each sensor assembly.
Only components belonging to *different* fingers are compared. The distance
check also sees visual geoms with contype/conaffinity zero, unlike data.contact.
"""

from __future__ import annotations

from itertools import combinations
import math

import mujoco
import numpy as np


FINGERS = ("finger1", "finger2", "finger3")


class SensorCollisionError(RuntimeError):
    """The current frame is unsafe and must not enter an accepted episode."""

    def __init__(self, report: dict):
        self.report = report
        pair = " / ".join(report.get("closest_pair") or [])
        super().__init__(
            f"Sensor clearance rejected: {report.get('reason', 'unsafe')}; "
            f"pair={pair}; distance_m={report.get('minimum_distance_m')}; "
            f"required_m={report.get('threshold_m')}"
        )


class SensorCollisionGuard:
    """Fail closed if the three sensor assemblies cannot be checked.

    ``inspect`` reads the positions already computed by mj_forward/mj_step; it
    does not advance or alter the simulation. A strictly positive separation is
    required even when clearance_m is zero. At a positive threshold, equality
    is accepted. Existing cross-finger MuJoCo contacts are rejected regardless
    of their margin, solver activation, or the geometric clearance threshold.
    """

    def __init__(self, model, clearance_m: float = 0.001):
        try:
            clearance_m = float(clearance_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("Sensor clearance_m must be finite and nonnegative") from exc
        if not math.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError("Sensor clearance_m must be finite and nonnegative")
        self.model = model
        self.clearance_m = clearance_m
        self.components: dict[int, tuple[str, str]] = {}
        missing = []
        for finger in FINGERS:
            for suffix in ("gel", "gelsight_housing"):
                name = f"{finger}_{suffix}"
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id < 0:
                    if suffix == "gel":
                        missing.append(name)
                    continue
                if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                    raise ValueError(f"Sensor component must be a box geom: {name}")
                self.components[geom_id] = (finger, name)
        if missing:
            raise ValueError("Missing required sensor gel geoms: " + ", ".join(missing))
        self.pairs = tuple(
            (first, second)
            for first, second in combinations(self.components, 2)
            if self.components[first][0] != self.components[second][0]
        )
        self.first_ids = np.array([pair[0] for pair in self.pairs], dtype=int)
        self.second_ids = np.array([pair[1] for pair in self.pairs], dtype=int)
        self.first_sizes = model.geom_size[self.first_ids].copy()
        self.second_sizes = model.geom_size[self.second_ids].copy()
        if not (np.isfinite(self.first_sizes).all() and np.isfinite(self.second_sizes).all()
                and (self.first_sizes > 0).all() and (self.second_sizes > 0).all()):
            raise ValueError("Sensor box sizes must be finite and greater than zero")

    def _base_report(self) -> dict:
        return {
            "safe": False,
            "reason": "unchecked",
            "minimum_distance_m": None,
            "threshold_m": self.clearance_m,
            "closest_pair": [],
            "closest_fingers": [],
            "cross_sensor_contacts": [],
            "checked_pairs": len(self.pairs),
            "distance_method": "obb_separating_axis_lower_bound",
            "distance_is_lower_bound": True,
        }

    def _separating_axis_gaps(self, data):
        """Return a conservative surface-gap bound for each oriented box pair.

        A positive gap on any separating axis proves the boxes are disjoint.
        Its largest value is a lower bound on Euclidean surface distance, so
        accepting only bounds >= clearance cannot accept an unsafe box pair.
        Negative values identify overlap. This avoids occasional zero-distance
        results from MuJoCo's generic box distance query on separated boxes.
        """
        first = data.geom_xmat[self.first_ids].reshape(-1, 3, 3)
        second = data.geom_xmat[self.second_ids].reshape(-1, 3, 3)
        basis_a = first.transpose(0, 2, 1)
        basis_b = second.transpose(0, 2, 1)
        cross = np.cross(basis_a[:, :, None, :], basis_b[:, None, :, :]).reshape(-1, 9, 3)
        axes = np.concatenate((basis_a, basis_b, cross), axis=1)
        lengths = np.linalg.norm(axes, axis=-1)
        valid = lengths > 1e-12
        axes = axes / np.where(valid, lengths, 1)[..., None]
        delta = data.geom_xpos[self.second_ids] - data.geom_xpos[self.first_ids]
        centers = np.abs(np.einsum("nkd,nd->nk", axes, delta))
        radius_a = np.einsum("nkj,nj->nk", np.abs(np.einsum("nkd,ndj->nkj", axes, first)), self.first_sizes)
        radius_b = np.einsum("nkj,nj->nk", np.abs(np.einsum("nkd,ndj->nkj", axes, second)), self.second_sizes)
        gaps = np.where(valid, centers - radius_a - radius_b, -np.inf)
        return gaps.max(axis=1)

    def inspect(self, data) -> dict:
        report = self._base_report()
        geom_ids = list(self.components)
        if not (
            np.isfinite(data.geom_xpos[geom_ids]).all()
            and np.isfinite(data.geom_xmat[geom_ids]).all()
        ):
            report["reason"] = "nonfinite_sensor_transform"
            return report

        gaps = self._separating_axis_gaps(data)
        if not np.isfinite(gaps).all():
            report["reason"] = "nonfinite_sensor_distance"
            return report
        closest = int(gaps.argmin())
        minimum = float(gaps[closest])
        pair = self.pairs[closest]
        report["minimum_distance_m"] = minimum
        report["closest_pair"] = [self.components[g][1] for g in pair]
        report["closest_fingers"] = [self.components[g][0] for g in pair]

        contacts = report["cross_sensor_contacts"]
        for contact in data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in self.components or second not in self.components:
                continue
            if self.components[first][0] == self.components[second][0]:
                continue
            distance = float(contact.dist)
            contacts.append({
                "pair": [self.components[g][1] for g in (first, second)],
                "fingers": [self.components[g][0] for g in (first, second)],
                "distance_m": distance if math.isfinite(distance) else None,
            })

        if contacts:
            report["reason"] = "cross_sensor_contact"
        elif minimum <= 0:
            report["reason"] = "sensor_surface_collision"
        elif minimum < self.clearance_m:
            report["reason"] = "sensor_clearance_below_threshold"
        else:
            report.update(safe=True, reason="ok")
        return report

    def require_safe(self, data) -> dict:
        report = self.inspect(data)
        if not report["safe"]:
            raise SensorCollisionError(report)
        return report
