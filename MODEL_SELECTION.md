# Semantic Model Selection

## Decision

The first OpenVision TensorRT bring-up pipeline uses:

| Stage | Model | Input | Output role |
|---|---|---|---|
| Person detection | RTMDet-nano | 320x320 | Bounded person regions |
| Whole-body pose | DWPose-m | 256x192 | 133 COCO-WholeBody SimCC joints |

This combination is selected because Lume needs body pointing and hand motion
from the same observation path. It is a bring-up baseline for engine/provider
implementation, not a production model decision for the ceiling viewpoint.

## Alternatives considered

| Model family | Strength | Reason not selected as the first manifest |
|---|---|---|
| RTMO | One-stage real-time multi-person pose | Standard published configuration exposes 17 body joints and does not satisfy the required hand vocabulary |
| RTMPose body models | Efficient top-down body pose | Body-only vocabulary does not support hand gesture observations |
| DWPose | Efficient whole-body pose with 133 joints | Selected; requires a separate detector and region-affine stage |

Official project/config evidence was inspected from:

- [MMPose RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [MMPose RTMO project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
- [MMPose DWPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [MMDeploy model support](https://github.com/open-mmlab/mmdeploy)
- [RTMDet paper](https://arxiv.org/abs/2212.07784)
- [RTMPose paper](https://arxiv.org/abs/2303.07399)
- [DWPose paper](https://arxiv.org/abs/2307.15880)
- [COCO-WholeBody paper](https://arxiv.org/abs/2007.11858)

Research snapshots used for this decision:

| Repository | Inspected revision |
|---|---|
| `open-mmlab/mmpose` | `759b39c13fea6ba094afc1fa932f51dc1b11cbf9` |
| `open-mmlab/mmdeploy` | `3f8604bd72e8e15d06b2e0552fe2fdb8f8de33c4` |
| `open-mmlab/mmdetection` | `cfd5d3a985b0249de009b67d04f37263e11cdf3d` |

## Checkpoint identity

| Model | SHA-256 |
|---|---|
| RTMDet-nano person detector | `05d8511e7b3fabc62e27d2f624179e004ad14ee63a86ca9d9d22c88f3db0eee1` |
| DWPose-m whole-body pose | `c8b7641988ce785c987b8ad156e411f328b54766d353df19aea5b08433ef1aaf` |

The checkpoint files are evidence inputs and are not stored in this repository.
Model caches must remain in the configured Hugging Face cache when a
Hugging Face distribution is used.

The pinned MMDeploy export produces `dets[1,N,5]` as `float32` and
`labels[1,N]` as `int64`, with `N` bounded to 100 by post-processing. The
semantic manifest preserves this runtime-variable dimension and element type;
it does not reinterpret the label tensor as a fixed-width `int32` array.

## Performance and memory implications

The detector runs once per frame. Pose cost scales with the bounded person
count, currently four. Full-frame RG10 conversion writes the detector tensor
directly on the GPU. Detector output remains on device, ROI metadata is compact,
and each pose input must be written by a dedicated region-affine GPU kernel.
No intermediate full-frame RGB image is permitted.

```text
RG10 host lease
  -> one H2D
    -> fused detector tensor
      -> detector engine
        -> compact device ROI list
          -> bounded region-affine pose tensors
            -> pose engine
              -> compact joint observations
```

## Product acceptance gates

1. Verify model and every training dataset license for the intended product
   distribution. Until then, manifest license identifiers remain absent.
2. Measure ceiling-view person recall, hand visibility, joint jitter,
   occlusion, and false positives using the real WAVESHARE-26185 camera.
3. Compare FP16 and FP32 against the source model with bounded joint-error
   fixtures before permitting an engine precision.
4. Meet the 33.3 ms p95 end-to-end budget with four-person worst-case input and
   zero frame-sized allocations after preparation.
5. Fine-tune or replace the baseline if ceiling-view accuracy fails; a
   successful engine conversion is not product-quality evidence.
