#!/bin/sh

set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
deployment_root="${repository_root}/Deployment/Wendy"
build_context="${deployment_root}/.build-context"

mkdir -p "${build_context}"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CTensorRTShim.cpp" \
    "${build_context}/CTensorRTShim.cpp"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/include/CTensorRTShim.h" \
    "${build_context}/CTensorRTShim.h"

shasum -a 256 \
    "${build_context}/CTensorRTShim.cpp" \
    "${build_context}/CTensorRTShim.h"
