import os

# Tests default to the offline configuration unless CI sets otherwise.
os.environ.setdefault("RADAR_MODE", "local")
os.environ.setdefault("RADAR_LLM_PROVIDER", "mock")
