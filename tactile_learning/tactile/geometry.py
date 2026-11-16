"""Sensor geometry from finger encoders only; no object pose or simulation contact truth."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .features import FEATURE_VERSION, FINGERS, atomic_save_npz, cache_is_valid, feature_path, read_signals, source_fingerprint


def xml_dependencies(xml: str | Path) -> list[Path]:
    """Track included kinematic XML files as well as the explicit root file."""
    seen = set()
    def visit(path):
        path = Path(path).resolve()
        if path in seen:
            return
        seen.add(path)
        tree = ET.parse(path)
        for include in tree.iter("include"):
            visit(path.parent / include.attrib["file"])
    visit(xml)
    return sorted(seen)


def edge_attributes(sensor_pose: np.ndarray) -> np.ndarray:
    """Directed i->j edge: p_j-p_i (m), distance (m), dot of local +Z normals."""
    positions = sensor_pose[..., :3]
    rotations = sensor_pose[..., 3:].reshape(*sensor_pose.shape[:-1], 3, 3)
    normals = rotations[..., :, 2]
    delta = positions[..., None, :, :] - positions[..., :, None, :]
    distances = np.linalg.norm(delta, axis=-1, keepdims=True)
    dot = np.sum(normals[..., :, None, :] * normals[..., None, :, :], axis=-1, keepdims=True)
    return np.concatenate((delta, distances, np.clip(dot, -1, 1)), axis=-1).astype(np.float32)


class GripperGeometry:
    def __init__(self, xml: str | Path, reference_body: str = "gripper_mount"):
        import mujoco
        self.mujoco = mujoco
        self.xml = Path(xml).resolve()
        self.dependencies = xml_dependencies(self.xml)
        self.reference_body = reference_body
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        self.body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, reference_body)
        self.sites = [self._id(mujoco.mjtObj.mjOBJ_SITE, f"{name}_gelsight_site") for name in FINGERS]
        joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{name}_hinge") for name in FINGERS]
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Finger encoder fields require scalar hinge joints")
        self.q_addresses = [int(self.model.jnt_qposadr[j]) for j in joints]

    def _id(self, kind, name):
        value = self.mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Required kinematic element missing from {self.xml}: {name}")
        return value

    def current_sensor_pose(self) -> np.ndarray:
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        origin = self.data.xpos[self.body_id]
        positions = (self.data.site_xpos[self.sites] - origin) @ rotation
        orientations = np.einsum("ab,nbc->nac", rotation.T, self.data.site_xmat[self.sites].reshape(3, 3, 3))
        return np.concatenate((positions, orientations.reshape(3, 9)), axis=-1).astype(np.float32)

    def replay(self, finger_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finger_q = np.asarray(finger_q, dtype=np.float64)
        if finger_q.ndim != 2 or finger_q.shape[1] != 3 or not len(finger_q) or not np.isfinite(finger_q).all():
            raise ValueError("Expected finite finger encoder angles [T,3] radians")
        self.mujoco.mj_resetData(self.model, self.data)
        poses = []
        for row in finger_q:
            self.data.qpos[self.q_addresses] = row
            self.mujoco.mj_forward(self.model, self.data)
            poses.append(self.current_sensor_pose())
        sensor_pose = np.stack(poses)
        return sensor_pose, edge_attributes(sensor_pose)

    def compute_episode(self, episode: dict, output: str | Path, force: bool = False) -> dict:
        signals = Path(episode["path"]) / "signals.csv"
        config = {"version": FEATURE_VERSION, "reference_body": self.reference_body, "normal_axis": "+Z",
                  "edge_direction": "i_to_j", "position_unit": "metre", "rotation_order": "row-major"}
        sources = [signals, *self.dependencies]
        fingerprint = source_fingerprint(sources, config)
        target = feature_path(output, "geometry", episode["id"])
        if not force and cache_is_valid(target, fingerprint):
            return {"id": episode["id"], "status": "cached", "path": str(target)}
        rows = read_signals(signals)
        angles = np.asarray([[float(row[f"q_f{i}"]) for i in (1, 2, 3)] for row in rows])
        sensor_pose, edge_attr = self.replay(angles)
        if fingerprint != source_fingerprint(sources, config):
            raise RuntimeError(f"Source changed during preprocessing: {signals}")
        metadata = {"schema_version": FEATURE_VERSION, "kind": "geometry", "source_fingerprint": fingerprint,
                    "backend": "mujoco_forward_kinematics", "config": config, "finger_order": list(FINGERS),
                    "sensor_pose": "site center and orientation relative to gripper body; not contact centroid",
                    "edge_attr": ["dx", "dy", "dz", "distance", "normal_dot"]}
        atomic_save_npz(target, metadata, sensor_pose=sensor_pose, edge_attr=edge_attr)
        return {"id": episode["id"], "status": "written", "path": str(target)}
