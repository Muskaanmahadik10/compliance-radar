import os
import shutil
from pathlib import Path

# Tests run offline and in their own state directory, so they never touch (or get blocked by) a running app:
# the embedded vector store allows one process per folder.
_STATE = Path(__file__).resolve().parent.parent / ".local-test"
shutil.rmtree(_STATE, ignore_errors=True)
os.environ["RADAR_MODE"] = os.environ.get("RADAR_MODE", "local")
os.environ["RADAR_LLM_PROVIDER"] = "mock"
os.environ["RADAR_LOCAL_QDRANT_PATH"] = str(_STATE / "qdrant")
os.environ["RADAR_LOCAL_SQLITE_PATH"] = str(_STATE / "radar.sqlite")
os.environ["RADAR_ARTIFACTS_DIR"] = str(_STATE / "artifacts")
