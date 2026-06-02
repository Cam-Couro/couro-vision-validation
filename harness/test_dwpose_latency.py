"""Measure DWPose-L per-frame latency at the only supported input
resolution (384x288). Records CoreML and CPU providers separately."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = (
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/"
    "multiview-validation/models/dw-ll_ucoco_384.onnx"
)
OUT_PATH = (
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/"
    "multiview-validation/data/layer2_higher_res/latency_baseline.json"
)


def bench(provider_list: list[str], h: int, w: int, n_warm: int = 3, n_iter: int = 30) -> float:
    sess = ort.InferenceSession(MODEL, providers=provider_list)
    x = np.random.randn(1, 3, h, w).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {"input": x})
    t0 = time.time()
    for _ in range(n_iter):
        sess.run(None, {"input": x})
    return (time.time() - t0) / n_iter * 1000.0


def main() -> None:
    results: dict[str, object] = {}
    for tag, providers in [
        ("coreml", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
        ("cpu", ["CPUExecutionProvider"]),
    ]:
        try:
            ms = bench(providers, 384, 288)
            print(f"{tag:8s} 384x288 batch=1  {ms:.2f} ms/frame")
            results[tag] = {"resolution": "384x288", "ms_per_frame": round(ms, 3)}
        except Exception as exc:  # noqa: BLE001
            print(f"{tag} failed: {exc}")
            results[tag] = {"error": str(exc)}

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
