FROM nvidia/cuda:13.2.0-base-ubuntu24.04@sha256:faef2047f4f69e613ea985d4d01575b2da75b5ca380ed98ed16c84843443e02e AS builder

ARG TENSORRT_ARCHIVE_SHA256=239bee69b7e8257b6add47f631c95a9e91cc07e0861365e9aa3f66a08bb166ae

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
        ca-certificates \
        cuda-crt-13-2 \
        cuda-cudart-dev-13-2 \
        curl \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN curl --fail --location --silent --show-error \
        https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.16.tar.gz \
        --output TensorRT-v10.16.tar.gz \
    && echo "${TENSORRT_ARCHIVE_SHA256}  TensorRT-v10.16.tar.gz" \
        | sha256sum --check \
    && tar --extract --gzip --file TensorRT-v10.16.tar.gz

COPY .build-context/CTensorRTShim.cpp ./CTensorRTShim.cpp
COPY .build-context/CTensorRTShim.h ./CTensorRTShim.h
COPY Probe.cpp ./Probe.cpp

RUN g++ \
        -std=c++17 \
        -O2 \
        -Wall \
        -Wextra \
        -Werror \
        -isystem /usr/local/cuda-13.2/include \
        -isystem TensorRT-10.16/include \
        -I . \
        CTensorRTShim.cpp \
        Probe.cpp \
        -ldl \
        -o openvision-tensorrt-runtime-probe

RUN g++ \
        -std=c++17 \
        -O1 \
        -g \
        -fno-omit-frame-pointer \
        -fsanitize=address,undefined \
        -Wall \
        -Wextra \
        -Werror \
        -isystem /usr/local/cuda-13.2/include \
        -isystem TensorRT-10.16/include \
        -I . \
        CTensorRTShim.cpp \
        Probe.cpp \
        -ldl \
        -o openvision-tensorrt-runtime-probe-sanitized

FROM ubuntu:24.04@sha256:7f622ca8766bccb22f04242ecb6f19f770b2f08827dc4b8c707de5e78a6da7ab

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
        libasan8 \
        libubsan1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder \
    /build/openvision-tensorrt-runtime-probe \
    /usr/local/bin/openvision-tensorrt-runtime-probe
COPY --from=builder \
    /build/openvision-tensorrt-runtime-probe-sanitized \
    /usr/local/bin/openvision-tensorrt-runtime-probe-sanitized

ENTRYPOINT ["/bin/sh", "-c", "ASAN_OPTIONS=detect_leaks=0 /usr/local/bin/openvision-tensorrt-runtime-probe-sanitized >/dev/null && exec /usr/local/bin/openvision-tensorrt-runtime-probe"]
