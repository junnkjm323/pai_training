# Same schema and topics as so101_features.py (state 6, target xyz 3, action 6); only the dataset names differ.
import importlib.util as _ilu
import os as _os

_spec = _ilu.spec_from_file_location("so101_features", _os.path.join(_os.path.dirname(__file__), "so101_features.py"))
_base = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_base)
globals().update({k: v for k, v in vars(_base).items() if not k.startswith("_")})

repo_id = "so101_wall"
root = "/home/ubuntu/pai2026-final-assignment/data/lerobot_dataset/so101_wall"
task_name = "reach over the wall"
