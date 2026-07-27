#!/bin/sh

set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
deployment_root="${repository_root}/Deployment/Wendy"
build_context="${deployment_root}/.build-context"

canonical_path() {
    source_path=$1
    if [ -d "${source_path}" ]; then
        CDPATH= cd -- "${source_path}" && pwd
        return
    fi
    source_directory=$(dirname -- "${source_path}")
    source_name=$(basename -- "${source_path}")
    printf '%s/%s\n' \
        "$(CDPATH= cd -- "${source_directory}" && pwd)" \
        "${source_name}"
}

reject_build_context_source() {
    variable_name=$1
    source_path=$2
    canonical_source=$(canonical_path "${source_path}")
    case "${canonical_source}" in
        "${build_context}"|"${build_context}"/*)
            printf '%s\n' \
                "${variable_name} must not reference the generated build context: ${canonical_source}" \
                >&2
            exit 1
            ;;
    esac
}

if [ -n "${OPENVISION_TRT_DETECTOR_PLAN:-}" ]; then
    reject_build_context_source \
        OPENVISION_TRT_DETECTOR_PLAN \
        "${OPENVISION_TRT_DETECTOR_PLAN}"
fi
if [ -n "${OPENVISION_TRT_POSE_PLAN:-}" ]; then
    reject_build_context_source \
        OPENVISION_TRT_POSE_PLAN \
        "${OPENVISION_TRT_POSE_PLAN}"
fi
if [ -n "${OPENVISION_TRT_RG10_FIXTURE:-}" ]; then
    reject_build_context_source \
        OPENVISION_TRT_RG10_FIXTURE \
        "${OPENVISION_TRT_RG10_FIXTURE}"
fi
if [ -n "${OPENVISION_TRT_EVALUATION_ROOT:-}" ]; then
    reject_build_context_source \
        OPENVISION_TRT_EVALUATION_ROOT \
        "${OPENVISION_TRT_EVALUATION_ROOT}"
fi
if [ -n "${OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT:-}" ]; then
    reject_build_context_source \
        OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT \
        "${OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT}"
fi

mkdir -p "${build_context}"
mkdir -p \
    "${build_context}/Evaluation" \
    "${build_context}/Fixtures" \
    "${build_context}/Models" \
    "${build_context}/Sources" \
    "${build_context}/Tests"
find "${build_context}/Fixtures" -type f -delete
find "${build_context}/Evaluation" -type f -delete
find "${build_context}/Models" -type f -delete
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CTensorRTShim.cpp" \
    "${build_context}/CTensorRTShim.cpp"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CRG10Preprocessor.cpp" \
    "${build_context}/CRG10Preprocessor.cpp"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CTensorRTEngine.cpp" \
    "${build_context}/CTensorRTEngine.cpp"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CPosePipeline.cpp" \
    "${build_context}/CPosePipeline.cpp"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/include/CTensorRTShim.h" \
    "${build_context}/CTensorRTShim.h"
install -m 0644 \
    "${repository_root}/Package.swift" \
    "${build_context}/Package.swift"
install -m 0644 \
    "${repository_root}/Package.resolved" \
    "${build_context}/Package.resolved"
rsync --archive --delete \
    "${repository_root}/Sources/" \
    "${build_context}/Sources/"
rsync --archive --delete \
    "${repository_root}/Tests/" \
    "${build_context}/Tests/"

if [ -n "${OPENVISION_TRT_DETECTOR_PLAN:-}" ]; then
    install -m 0644 \
        "${OPENVISION_TRT_DETECTOR_PLAN}" \
        "${build_context}/Models/rtmdet-fp16.plan"
fi
if [ -n "${OPENVISION_TRT_POSE_PLAN:-}" ]; then
    install -m 0644 \
        "${OPENVISION_TRT_POSE_PLAN}" \
        "${build_context}/Models/dwpose-fp16.plan"
fi
if [ -n "${OPENVISION_TRT_RG10_FIXTURE:-}" ]; then
    install -m 0644 \
        "${OPENVISION_TRT_RG10_FIXTURE}" \
        "${build_context}/Fixtures/person.rg10"
fi
if [ -n "${OPENVISION_TRT_EVALUATION_ROOT:-}" ]; then
    install -m 0644 \
        "${OPENVISION_TRT_EVALUATION_ROOT}/runtime-manifest.tsv" \
        "${build_context}/Evaluation/runtime-manifest.tsv"
    find "${OPENVISION_TRT_EVALUATION_ROOT}/fixtures" \
        -type f \
        -name '*.rg10' \
        -exec install -m 0644 {} "${build_context}/Evaluation" \;
fi
if [ -n "${OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT:-}" ]; then
    install -m 0644 \
        "${OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT}/runtime-manifest.tsv" \
        "${build_context}/Evaluation/temporal-runtime-manifest.tsv"
    find "${OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT}/fixtures" \
        -type f \
        -name '*.rg10' \
        -exec install -m 0644 {} "${build_context}/Evaluation" \;
fi

shasum -a 256 \
    "${build_context}/CTensorRTShim.cpp" \
    "${build_context}/CRG10Preprocessor.cpp" \
    "${build_context}/CTensorRTEngine.cpp" \
    "${build_context}/CPosePipeline.cpp" \
    "${build_context}/CTensorRTShim.h" \
    "${build_context}/Package.swift" \
    "${build_context}/Package.resolved"

if find "${build_context}/Models" -type f -name '*.plan' | grep -q .; then
    shasum -a 256 "${build_context}/Models"/*.plan
fi
if find "${build_context}/Fixtures" -type f -name '*.rg10' | grep -q .; then
    shasum -a 256 "${build_context}/Fixtures"/*.rg10
fi
if find "${build_context}/Evaluation" -type f -name '*.rg10' | grep -q .; then
    shasum -a 256 "${build_context}/Evaluation"/*.rg10
fi
