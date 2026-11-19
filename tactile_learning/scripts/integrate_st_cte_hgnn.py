
from pathlib import Path
import ast, shutil

ROOT = Path.cwd()
CTE = ROOT / "tactile/models/cte_hgnn.py"
FACTORY = ROOT / "tactile/models/classifier.py"
CONFIG = ROOT / "tactile/config.py"
for p in (CTE, FACTORY, CONFIG):
    if not p.exists():
        raise SystemExit(f"Run from tactile_learning root; missing {p}")

def backup(p):
    b = p.with_name(p.name + ".bak_before_st_cte")
    if not b.exists():
        shutil.copy2(p, b)
        print("[BACKUP]", b)

def replace_once(s, old, new, label):
    count = s.count(old)
    if count == 0 and new in s:
        print("[SKIP]", label, "already applied")
        return s
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, got {count}; no write made")
    return s.replace(old, new, 1)

# Confirm the new modules exist before modifying anything.
for p in (ROOT / "tactile/models/spatial_graph.py",
          ROOT / "tactile/models/st_cte_model.py"):
    if not p.exists():
        raise SystemExit(f"Missing model file: {p}")
    ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))

# CTE temporal embedding output; old API remains default.
s = CTE.read_text(encoding="utf-8-sig")
s = replace_once(
    s,
    "        query_time=None,\n        return_aux=True,\n    ):",
    "        query_time=None,\n        return_aux=True,\n        return_features=False,\n    ):",
    "CTE forward signature",
)
old = (
    '        logits = self.head(pooled)\n\n'
    '        if not return_aux:\n'
    '            return logits\n\n'
    '        aux = {\n'
    '            **ro.losses,\n'
    '            **{f"flow_{k}": v for k, v in fo.losses.items()},\n'
    '            **route_aux,\n'
    '        }\n\n'
    '        return logits, aux\n'
)
new = (
    '        logits = self.head(pooled)\n'
    '        aux = {\n'
    '            **ro.losses,\n'
    '            **{f"flow_{k}": v for k, v in fo.losses.items()},\n'
    '            **route_aux,\n'
    '        }\n'
    '        if return_features:\n'
    '            return (pooled, aux) if return_aux else pooled\n'
    '        if not return_aux:\n'
    '            return logits\n'
    '        return logits, aux\n'
)
s = replace_once(s, old, new, "CTE output block")
ast.parse(s, filename=str(CTE))
backup(CTE)
CTE.write_text(s, encoding="utf-8", newline="\n")

# Model factory route.
s = FACTORY.read_text(encoding="utf-8-sig")
old = (
    '    if architecture == "cte_hgnn":\n'
    '        from .cte_model import CTETactileClassifier\n'
    '        return CTETactileClassifier(num_classes, config)\n\n'
    '    if architecture != "baseline":\n'
)
new = (
    '    if architecture == "cte_hgnn":\n'
    '        from .cte_model import CTETactileClassifier\n'
    '        return CTETactileClassifier(num_classes, config)\n\n'
    '    if architecture == "st_cte_hgnn":\n'
    '        from .st_cte_model import STCTETactileClassifier\n'
    '        return STCTETactileClassifier(num_classes, config)\n\n'
    '    if architecture != "baseline":\n'
)
s = replace_once(s, old, new, "classifier factory")
ast.parse(s, filename=str(FACTORY))
backup(FACTORY)
FACTORY.write_text(s, encoding="utf-8", newline="\n")

# Configuration defaults, variant, and validation.
s = CONFIG.read_text(encoding="utf-8-sig")
s = replace_once(
    s,
    '        "cte_sigma_seconds": 0.03,\n',
    '        "cte_sigma_seconds": 0.03,\n'
    '        "spatial_dim": 128,\n'
    '        "spatial_graph_layers": 2,\n',
    "spatial defaults",
)
s = replace_once(
    s,
    '    "cte_hgnn": {\n',
    '    "st_cte_hgnn": {\n'
    '        "model": {\n'
    '            "architecture": "st_cte_hgnn",\n'
    '            "use_flow": True,\n'
    '            "use_alignment": False,\n'
    '            "fusion": "graph",\n'
    '            "use_edge_geometry": False,\n'
    '            "use_local_graph": False,\n'
    '        },\n'
    '        "training": {"temporal_shuffle": False},\n'
    '    },\n'
    '    "cte_hgnn": {\n',
    "ST-CTE variant",
)
s = replace_once(
    s,
    'if model.get("architecture", "baseline") not in ("baseline", "cte_hgnn"):',
    'if model.get("architecture", "baseline") not in ("baseline", "cte_hgnn", "st_cte_hgnn"):',
    "architecture validation",
)
ast.parse(s, filename=str(CONFIG))
backup(CONFIG)
CONFIG.write_text(s, encoding="utf-8", newline="\n")

for p in (CTE, FACTORY, CONFIG,
          ROOT / "tactile/models/spatial_graph.py",
          ROOT / "tactile/models/st_cte_model.py"):
    ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))

print("[PASS] integration patch applied; Python syntax valid")
print("Next: python -m pytest tests/test_st_cte_gradients.py -q")
