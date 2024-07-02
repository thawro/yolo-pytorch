"""Config constants"""

from pathlib import Path

from src.utils.utils import get_current_date_and_time

ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = ROOT / "results"
YAML_EXP_PATH = ROOT / "experiments"
DS_ROOT = ROOT / "data"

INFERENCE_OUT_PATH = ROOT / "inference_out"
INFERENCE_OUT_PATH.mkdir(exist_ok=True, parents=True)

NOW = get_current_date_and_time()
