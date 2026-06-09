"""Agent TT: dump v44 per-slot predictions into the SS unified format."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.per_slot_prediction_dump import DumpAccumulator, DUMP_DIR
from harness import layer3_retrain_on_videopose3d_l2 as tt


def main() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DUMP_DIR / "per_slot_predictions_v44.json"
    if out_path.exists():
        print(f"[TT-dump] v44 already dumped at {out_path}; overwriting.")
    print(f"[TT-dump] === dumping v44 (VideoPose3D L2 + L3 ridge) ===")
    t0 = time.time()
    tt.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v44",
        loso_discipline=(
            "Layer-3-LOSO (clean) -- VideoPose3D L2 pretrained on "
            "Human3.6M (disjoint from OpenCap and ASPset)"
        ),
    )
    tt.run()
    written = tt.DUMP_ACCUMULATOR.write(out_path)
    tt.DUMP_ACCUMULATOR = None
    print(f"[TT-dump] wrote {written} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
