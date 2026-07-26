# Model Export Contract

This directory reproduces the backend-neutral ONNX inputs used to build
OpenVision TensorRT engines. Checkpoints, ONNX files, and TensorRT plans are
external artifacts and must not be committed.

## Pinned sources

| Source | Revision |
|---|---|
| `open-mmlab/mmpose` | `759b39c13fea6ba094afc1fa932f51dc1b11cbf9` |
| `open-mmlab/mmdeploy` | `3f8604bd72e8e15d06b2e0552fe2fdb8f8de33c4` |
| `open-mmlab/mmdetection` | `cfd5d3a985b0249de009b67d04f37263e11cdf3d` |

| Checkpoint | SHA-256 |
|---|---|
| RTMDet-nano person detector | `05d8511e7b3fabc62e27d2f624179e004ad14ee63a86ca9d9d22c88f3db0eee1` |
| DWPose-trained RTMPose-m whole-body pose | `c8b7641988ce785c987b8ad156e411f328b54766d353df19aea5b08433ef1aaf` |

The pose checkpoint is distributed by the MMPose RTMPose project as a
COCO-WholeBody plus UBody DWPose model. Export uses the matching
`TopdownPoseEstimator` RTMPose-m inference architecture rather than the
training-only `DWPoseDistiller` wrapper.

## Environment

The export Python environment must contain compatible PyTorch, MMEngine,
MMCV operations, ONNX, ONNX Runtime, NumPy, and the MMPose runtime
dependencies. The script verifies repository revisions and checkpoint
digests before allowing the legacy checkpoint loader.

```bash
export OPENVISION_EXPORT_PYTHON=/path/to/venv/bin/python
export OPENVISION_MMDEPLOY_ROOT=/path/to/mmdeploy
export OPENVISION_MMPOSE_ROOT=/path/to/mmpose
export OPENVISION_MMDET_ROOT=/path/to/mmdetection
export OPENVISION_RTMD_CHECKPOINT=/path/to/rtmdet.pth
export OPENVISION_POSE_CHECKPOINT=/path/to/pose.pth
export OPENVISION_EXPORT_ROOT=/temporary/output
export OPENVISION_EXTRA_PYTHONPATH=/path/to/torch/site-packages

./Deployment/ModelExport/export-onnx.sh
```

`OPENVISION_EXTRA_PYTHONPATH` is optional when PyTorch is installed in the
export environment itself.

## Verified ONNX contract

| Stage | Input | Outputs |
|---|---|---|
| RTMDet-nano | `float32 [1,3,320,320]` | `dets float32 [1,N,5]`, `labels int64 [1,N]`, `N <= 100` |
| Whole-body pose | `float32 [batch,3,256,192]` | `simcc_x float32 [batch,133,384]`, `simcc_y float32 [batch,133,512]` |

The verifier runs ONNX checker and ONNX Runtime behavior for the detector and
pose batches one and four. This verifies the exchange graph, not TensorRT
compatibility or model accuracy.
