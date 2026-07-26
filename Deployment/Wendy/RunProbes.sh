#!/bin/sh

set -eu

detector_plan=/opt/openvision/models/rtmdet-fp16.plan
pose_plan=/opt/openvision/models/dwpose-fp16.plan
person_fixture=/opt/openvision/fixtures/person.rg10

ASAN_OPTIONS=detect_leaks=0 \
    /usr/local/bin/openvision-tensorrt-runtime-probe-sanitized \
    >/dev/null

c_result=$(/usr/local/bin/openvision-tensorrt-runtime-probe)

if [ -f "${detector_plan}" ] && [ -f "${pose_plan}" ]; then
    swift_result=$(
        /usr/local/bin/openvision-tensorrt-swift-probe \
            "${detector_plan}" \
            "${OPENVISION_DETECTOR_PLAN_SHA256}"
    )
    engine_result=$(
        /usr/local/bin/openvision-tensorrt-engine-probe \
            "${detector_plan}" \
            "${OPENVISION_DETECTOR_PLAN_SHA256}" \
            "${pose_plan}" \
            "${OPENVISION_POSE_PLAN_SHA256}"
    )
    checksum_failure_result=$(
        /usr/local/bin/openvision-tensorrt-engine-probe \
            --verify-checksum-failure \
            "${detector_plan}" \
            0000000000000000000000000000000000000000000000000000000000000000
    )
    semantic_mismatch_result=$(
        /usr/local/bin/openvision-tensorrt-engine-probe \
            --verify-semantic-mismatch \
            "${detector_plan}" \
            "${OPENVISION_DETECTOR_PLAN_SHA256}"
    )
    if [ -f "${person_fixture}" ]; then
        provider_result=$(
            /usr/local/bin/openvision-tensorrt-swift-probe \
                --provider \
                "${detector_plan}" \
                "${OPENVISION_DETECTOR_PLAN_SHA256}" \
                "${pose_plan}" \
                "${OPENVISION_POSE_PLAN_SHA256}" \
                "${person_fixture}"
        )
    else
        provider_result='{"status":"notRequested","reason":"missingFixture"}'
    fi
elif [ ! -f "${detector_plan}" ] && [ ! -f "${pose_plan}" ]; then
    swift_result=$(
        /usr/local/bin/openvision-tensorrt-swift-probe
    )
    engine_result='{"status":"notRequested"}'
    checksum_failure_result='{"status":"notRequested"}'
    semantic_mismatch_result='{"status":"notRequested"}'
    provider_result='{"status":"notRequested"}'
else
    printf '%s\n' \
        'TensorRT engine probe requires both detector and pose plans.' \
        >&2
    exit 1
fi

printf '{"cProbe":%s,"swiftProbe":%s,"engineProbe":%s,"checksumFailureProbe":%s,"semanticMismatchProbe":%s,"providerProbe":%s}\n' \
    "${c_result}" \
    "${swift_result}" \
    "${engine_result}" \
    "${checksum_failure_result}" \
    "${semantic_mismatch_result}" \
    "${provider_result}"
