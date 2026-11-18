# Event-Driven Three-Finger GelSight Collector V2

## Included files
- `collector.py`: MCB/legacy collector adapter and event-driven loop.
- `core.py`: state machine, per-finger controller primitives, contact tracker,
  event buffer and quality checks.
- `tests/test_core.py`: pure-Python unit tests.

## Machine-specific model
Use the included `clamp_train_00015149_v2.xml` from the companion download.
It resolves the existing assets relative to the project:
- UR3/gripper STL: `simulation/robot/xml/`
- MCB OBJ: `data/raw/MCB_A/dataset_org_norm/train/Clamps/00015149.obj`

The object retains `mcb_part_free`, zero gravity, and is not pinned.

## Important controller detail
The MCB adapter has three separate position actuators:
`finger1_ctrl`, `finger2_ctrl`, `finger3_ctrl`.
V2 updates their targets independently. A finger currently in contact holds
its own target while non-contacting fingers continue closing. The scalar
schedule (0.35 / 0.12 / 0.06 / 0.00) is a legacy command-coordinate rate,
not measured physical hinge velocity.

## Install
Extract `event_driven_v2` into:
`tactile_learning/scripts/collection/event_driven_v2/`

Keep the original collector files and `sensor_collision_guard.py` unchanged.

## Test
From `tactile_learning`:
```powershell
python -m unittest discover -s scripts/collection/event_driven_v2/tests -v
```

## XML preflight
```powershell
python scripts/collection/event_driven_v2/collector.py `
  --xml simulation/mcb_xml/clamp_train_00015149_v2.xml `
  --check-only
```

## 5-episode pilot
Only after preflight passes:
```powershell
python scripts/collection/event_driven_v2/collector.py `
  --xml simulation/mcb_xml/clamp_train_00015149_v2.xml `
  --output "E:/project/p1/deepgelsight/event_driven_v2" `
  --episodes 5 --seed 20261118 --show-viewer --run
```

Output is separate from the existing 68-category dataset.


## TCP and object-centre positioning (explicit V2 contract)

V2 preserves the positioning path from the existing collector:
- TCP site: `tool0`
- Tool-frame TCP-to-nominal-grasp-centre offset: `[0.14965, 0, 0]` m
- Object-frame nominal grasp centre: `GRASP_CENTER_OBJECT` from the legacy collector
- Per-episode object-frame offset: `pose_config["grasp_offset_object_m"]`

The legacy run loop computes the target grasp centre in world coordinates
from the settled object's position and rotation, then subtracts the rotated
TCP offset to obtain the target TCP position. V2 does not replace this with
the object's world origin or a guessed bounding-box centre. The values are
also written into each episode's metadata for auditability.
