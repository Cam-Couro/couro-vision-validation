#!/bin/bash
# Run bbox + GT + DWPose + pred + correlate for one S? Seq1 cam7 (rear).
set -e
RA="/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation"
S="$1"

cd "$RA"
echo "=== $S Seq1 cam7 (rear) ==="
python3 -m harness.mpi3dhp_bboxes "$RA/data/mpi_inf_3dhp/$S/Seq1/annot.mat" 7 "$RA/data/mpi_inf_3dhp_dwpose/${S}_Seq1_cam7_boxes.csv" 2>&1 | tail -1
python3 -m harness.mpi3dhp_gt_angles "$RA/data/mpi_inf_3dhp/$S/Seq1/annot.mat" "$RA/data/mpi_inf_3dhp_gt/${S}_Seq1_angles.json" 2>&1 | grep -E "(Wrote|hip_flexion_r)" | head -2
python3 -m harness.mpi3dhp_infer_dwpose "$RA/data/mpi_inf_3dhp/$S/Seq1/imageSequence/video_7.avi" "$RA/data/mpi_inf_3dhp_dwpose/${S}_Seq1_cam7_boxes.csv" "$RA/data/mpi_inf_3dhp_dwpose/${S}_Seq1_cam7.json" --batch 8 2>&1 | tail -1
python3 -m harness.mpi3dhp_pred_angles "$RA/data/mpi_inf_3dhp_dwpose/${S}_Seq1_cam7.json" "$RA/data/mpi_inf_3dhp_dwpose/${S}_Seq1_cam7_pred_angles.json" 2>&1 | tail -1
python3 -m harness.mpi3dhp_correlate "$RA/data/mpi_inf_3dhp_dwpose/${S}_Seq1_cam7_pred_angles.json" "$RA/data/mpi_inf_3dhp_gt/${S}_Seq1_angles.json" --label "${S}_Seq1_cam7" --out-json "$RA/data/mpi_inf_3dhp_rear_validation/r_${S}_Seq1_cam7.json" 2>&1
