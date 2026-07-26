#!/bin/sh

set -eu

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

required_variable() {
    variable_name=$1
    eval "variable_value=\${${variable_name}:-}"
    if [ -z "${variable_value}" ]; then
        echo "Missing required environment variable: ${variable_name}" >&2
        exit 2
    fi
}

required_variable OPENVISION_EXPORT_PYTHON
required_variable OPENVISION_MMDEPLOY_ROOT
required_variable OPENVISION_MMPOSE_ROOT
required_variable OPENVISION_MMDET_ROOT
required_variable OPENVISION_RTMD_CHECKPOINT
required_variable OPENVISION_POSE_CHECKPOINT
required_variable OPENVISION_EXPORT_ROOT

MMPOSE_REVISION=759b39c13fea6ba094afc1fa932f51dc1b11cbf9
MMDEPLOY_REVISION=3f8604bd72e8e15d06b2e0552fe2fdb8f8de33c4
MMDET_REVISION=cfd5d3a985b0249de009b67d04f37263e11cdf3d
RTMDET_CHECKSUM=05d8511e7b3fabc62e27d2f624179e004ad14ee63a86ca9d9d22c88f3db0eee1
POSE_CHECKSUM=c8b7641988ce785c987b8ad156e411f328b54766d353df19aea5b08433ef1aaf

verify_revision() {
    repository=$1
    expected=$2
    actual=$(git -C "${repository}" rev-parse HEAD)
    if [ "${actual}" != "${expected}" ]; then
        echo "Repository revision mismatch: ${repository}" >&2
        echo "Expected: ${expected}" >&2
        echo "Actual:   ${actual}" >&2
        exit 3
    fi
}

verify_checksum() {
    artifact=$1
    expected=$2
    actual=$(shasum -a 256 "${artifact}" | awk '{print $1}')
    if [ "${actual}" != "${expected}" ]; then
        echo "Checkpoint checksum mismatch: ${artifact}" >&2
        echo "Expected: ${expected}" >&2
        echo "Actual:   ${actual}" >&2
        exit 4
    fi
}

verify_revision "${OPENVISION_MMPOSE_ROOT}" "${MMPOSE_REVISION}"
verify_revision "${OPENVISION_MMDEPLOY_ROOT}" "${MMDEPLOY_REVISION}"
verify_revision "${OPENVISION_MMDET_ROOT}" "${MMDET_REVISION}"
verify_checksum "${OPENVISION_RTMD_CHECKPOINT}" "${RTMDET_CHECKSUM}"
verify_checksum "${OPENVISION_POSE_CHECKPOINT}" "${POSE_CHECKSUM}"

mkdir -p \
    "${OPENVISION_EXPORT_ROOT}/rtmdet" \
    "${OPENVISION_EXPORT_ROOT}/pose"

temporary_directory=$(
    mktemp -d "${TMPDIR:-/tmp}/openvision-rtmdet-export.XXXXXX"
)
temporary_config="${temporary_directory}/deploy.py"
cleanup() {
    rm -f "${temporary_config}"
    rmdir "${temporary_directory}"
}
trap cleanup EXIT HUP INT TERM

printf '%s\n' \
    "_base_ = [\"${OPENVISION_MMDEPLOY_ROOT}/configs/mmdet/detection/detection_onnxruntime_static.py\"]" \
    'onnx_config = dict(input_shape=(320, 320))' \
    > "${temporary_config}"

python_path="${OPENVISION_MMDET_ROOT}:${OPENVISION_MMPOSE_ROOT}:${OPENVISION_MMDEPLOY_ROOT}"
if [ -n "${OPENVISION_EXTRA_PYTHONPATH:-}" ]; then
    python_path="${python_path}:${OPENVISION_EXTRA_PYTHONPATH}"
fi

demo_image="${OPENVISION_MMDEPLOY_ROOT}/demo/resources/human-pose.jpg"
export_tool="${OPENVISION_MMDEPLOY_ROOT}/tools/torch2onnx.py"
detector_config="${OPENVISION_MMPOSE_ROOT}/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py"
pose_config="${OPENVISION_MMPOSE_ROOT}/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"
pose_deploy_config="${OPENVISION_MMDEPLOY_ROOT}/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py"

(
    cd "${OPENVISION_MMPOSE_ROOT}"
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH="${python_path}" \
    "${OPENVISION_EXPORT_PYTHON}" "${export_tool}" \
        "${temporary_config}" \
        "${detector_config}" \
        "${OPENVISION_RTMD_CHECKPOINT}" \
        "${demo_image}" \
        --work-dir "${OPENVISION_EXPORT_ROOT}/rtmdet" \
        --device cpu

    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH="${python_path}" \
    "${OPENVISION_EXPORT_PYTHON}" "${export_tool}" \
        "${pose_deploy_config}" \
        "${pose_config}" \
        "${OPENVISION_POSE_CHECKPOINT}" \
        "${demo_image}" \
        --work-dir "${OPENVISION_EXPORT_ROOT}/pose" \
        --device cpu
)

"${OPENVISION_EXPORT_PYTHON}" \
    "${script_root}/verify-onnx.py" \
    --detector "${OPENVISION_EXPORT_ROOT}/rtmdet/end2end.onnx" \
    --pose "${OPENVISION_EXPORT_ROOT}/pose/end2end.onnx"
