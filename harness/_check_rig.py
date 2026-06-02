"""Quick sanity check: do all subjects share the same studio rig calibration?"""
import sys
from pathlib import Path
HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
from mpi3dhp_camera_analysis import classify_cameras, parse_calibration

ROOT = HARNESS.parent / "data" / "mpi_inf_3dhp"
for s in range(1, 9):
    calib = ROOT / f"S{s}" / "Seq2" / "camera.calibration"
    if not calib.exists():
        print(f"S{s}: MISSING")
        continue
    cams = parse_calibration(calib)
    layout = classify_cameras(cams)
    cam7 = next((c for c in layout["cameras"] if c["cam_id"] == 7), None)
    if cam7 is None:
        print(f"S{s}: cam 7 not in calibration")
    else:
        print(
            f"S{s}: cam7 rel_az={cam7['rel_az_from_cam0']:+.1f}deg "
            f"height={cam7['height_mm']:.0f}mm radius={cam7['radius_mm']:.0f}mm "
            f"category={cam7['category']}"
        )
