"""Configuracoes estaveis e caminhos padrao do projeto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "artifacts" / "baseline_results.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "baseline.log"

RANDOM_STATE = 42
TEST_SIZE = 0.20
PHASE1_VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.25
EXPECTED_DATASET_SHA256 = (
    "1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33"
)

