"""Generate the company and load it into the database: python -m radar.data.build"""

from __future__ import annotations

from radar.config import DATA_DIR
from radar.data.generate import write_all
from radar.db import load_company


def build_all(seed: int = 7) -> dict:
    meta = write_all(DATA_DIR / "generated", seed)
    loaded = load_company(DATA_DIR / "generated")
    print({"generated": meta, "loaded": loaded})
    return meta


if __name__ == "__main__":
    build_all()
