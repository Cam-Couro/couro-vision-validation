"""Test whether DWPose-L ONNX accepts variable spatial input shapes.

Per onnxruntime metadata the input is declared ['batch', 3, 384, 288] — only
batch is dynamic. We empirically confirm by trying alternate H,W combos.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

MODEL_PATH = (
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/"
    "multiview-validation/models/dw-ll_ucoco_384.onnx"
)


def main() -> None:
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    decl = sess.get_inputs()[0].shape
    print(f"Declared input shape: {decl}")

    sizes = [(384, 288), (480, 352), (576, 432), (768, 576)]
    for h, w in sizes:
        x = np.zeros((1, 3, h, w), dtype=np.float32)
        try:
            out = sess.run(None, {"input": x})
            print(f"OK  {h}x{w}: simcc_x={out[0].shape} simcc_y={out[1].shape}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERR {h}x{w}: {exc}")
            print("---")


if __name__ == "__main__":
    main()
