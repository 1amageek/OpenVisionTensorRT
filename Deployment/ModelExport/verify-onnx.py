import argparse
import hashlib
import json
from pathlib import Path

import numpy
import onnx
import onnxruntime


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)


def verify_detector(path: Path) -> dict[str, object]:
    verify_model(path)
    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = {output.name: output for output in session.get_outputs()}
    if len(inputs) != 1:
        raise ValueError("Detector must expose exactly one input")
    if inputs[0].name != "input":
        raise ValueError("Detector input must be named input")
    if inputs[0].shape != [1, 3, 320, 320]:
        raise ValueError(f"Unexpected detector input shape: {inputs[0].shape}")
    if inputs[0].type != "tensor(float)":
        raise ValueError(f"Unexpected detector input type: {inputs[0].type}")
    if set(outputs) != {"dets", "labels"}:
        raise ValueError(f"Unexpected detector outputs: {sorted(outputs)}")
    if outputs["dets"].type != "tensor(float)":
        raise ValueError(f"Unexpected dets type: {outputs['dets'].type}")
    if outputs["labels"].type != "tensor(int64)":
        raise ValueError(f"Unexpected labels type: {outputs['labels'].type}")

    detections, labels = session.run(
        ["dets", "labels"],
        {"input": numpy.zeros((1, 3, 320, 320), dtype=numpy.float32)},
    )
    if detections.ndim != 3 or detections.shape[0] != 1:
        raise ValueError(f"Unexpected dets shape: {detections.shape}")
    if detections.shape[2] != 5 or detections.shape[1] > 100:
        raise ValueError(f"Unexpected dets bounds: {detections.shape}")
    if labels.shape != detections.shape[:2]:
        raise ValueError(
            f"Detector output count mismatch: {detections.shape}, {labels.shape}"
        )
    if detections.dtype != numpy.float32 or labels.dtype != numpy.int64:
        raise ValueError(
            f"Unexpected detector runtime types: {detections.dtype}, {labels.dtype}"
        )
    if not numpy.isfinite(detections).all():
        raise ValueError("Detector produced non-finite values")

    return {
        "sha256": checksum(path),
        "detsShape": list(detections.shape),
        "labelsShape": list(labels.shape),
        "labelsType": str(labels.dtype),
    }


def verify_pose(path: Path) -> dict[str, object]:
    verify_model(path)
    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = {output.name: output for output in session.get_outputs()}
    if len(inputs) != 1:
        raise ValueError("Pose model must expose exactly one input")
    if inputs[0].name != "input":
        raise ValueError("Pose input must be named input")
    if inputs[0].shape != ["batch", 3, 256, 192]:
        raise ValueError(f"Unexpected pose input shape: {inputs[0].shape}")
    if inputs[0].type != "tensor(float)":
        raise ValueError(f"Unexpected pose input type: {inputs[0].type}")
    if set(outputs) != {"simcc_x", "simcc_y"}:
        raise ValueError(f"Unexpected pose outputs: {sorted(outputs)}")
    if any(output.type != "tensor(float)" for output in outputs.values()):
        raise ValueError("Pose outputs must use float tensors")

    runtime_shapes: dict[str, list[list[int]]] = {}
    for batch in (1, 4):
        simcc_x, simcc_y = session.run(
            ["simcc_x", "simcc_y"],
            {
                "input": numpy.zeros(
                    (batch, 3, 256, 192),
                    dtype=numpy.float32,
                )
            },
        )
        if simcc_x.shape != (batch, 133, 384):
            raise ValueError(f"Unexpected simcc_x shape: {simcc_x.shape}")
        if simcc_y.shape != (batch, 133, 512):
            raise ValueError(f"Unexpected simcc_y shape: {simcc_y.shape}")
        if simcc_x.dtype != numpy.float32 or simcc_y.dtype != numpy.float32:
            raise ValueError(
                f"Unexpected pose runtime types: {simcc_x.dtype}, {simcc_y.dtype}"
            )
        if not numpy.isfinite(simcc_x).all():
            raise ValueError("simcc_x produced non-finite values")
        if not numpy.isfinite(simcc_y).all():
            raise ValueError("simcc_y produced non-finite values")
        runtime_shapes[str(batch)] = [
            list(simcc_x.shape),
            list(simcc_y.shape),
        ]

    return {
        "sha256": checksum(path),
        "runtimeShapes": runtime_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    arguments = parser.parse_args()

    result = {
        "detector": verify_detector(arguments.detector),
        "pose": verify_pose(arguments.pose),
    }
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
