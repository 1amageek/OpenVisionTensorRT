#!/bin/sh

set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
deployment_root="${repository_root}/Deployment/Wendy"
build_context="${deployment_root}/.build-context"

mkdir -p "${build_context}"
mkdir -p \
    "${build_context}/Sources" \
    "${build_context}/Tests"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CTensorRTShim.cpp" \
    "${build_context}/CTensorRTShim.cpp"
install -m 0644 \
    "${repository_root}/Sources/CTensorRTShim/CRG10Preprocessor.cpp" \
    "${build_context}/CRG10Preprocessor.cpp"
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

shasum -a 256 \
    "${build_context}/CTensorRTShim.cpp" \
    "${build_context}/CRG10Preprocessor.cpp" \
    "${build_context}/CTensorRTShim.h" \
    "${build_context}/Package.swift" \
    "${build_context}/Package.resolved"
