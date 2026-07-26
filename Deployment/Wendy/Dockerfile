FROM nvidia/cuda:13.2.0-base-ubuntu24.04@sha256:faef2047f4f69e613ea985d4d01575b2da75b5ca380ed98ed16c84843443e02e AS builder

ARG TENSORRT_ARCHIVE_SHA256=239bee69b7e8257b6add47f631c95a9e91cc07e0861365e9aa3f66a08bb166ae
ARG SWIFT_SNAPSHOT=swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-17-a

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
        binutils \
        ca-certificates \
        clang \
        cuda-crt-13-2 \
        cuda-cudart-dev-13-2 \
        cuda-nvrtc-dev-13-2 \
        curl \
        g++ \
        git \
        libc6-dev \
        libcurl4-openssl-dev \
        libedit2 \
        libgcc-13-dev \
        libncurses6 \
        libpython3.12 \
        libsqlite3-0 \
        libstdc++-13-dev \
        libxml2-dev \
        libz3-dev \
        pkg-config \
        tzdata \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN curl --fail --location --silent --show-error \
        https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.16.tar.gz \
        --output TensorRT-v10.16.tar.gz \
    && echo "${TENSORRT_ARCHIVE_SHA256}  TensorRT-v10.16.tar.gz" \
        | sha256sum --check \
    && tar --extract --gzip --file TensorRT-v10.16.tar.gz

RUN curl --fail --location --silent --show-error \
        "https://download.swift.org/swift-6.4.x-branch/ubuntu2404-aarch64/${SWIFT_SNAPSHOT}/${SWIFT_SNAPSHOT}-ubuntu24.04-aarch64.tar.gz" \
        --output swift-toolchain.tar.gz \
    && mkdir -p /opt/swift \
    && tar \
        --extract \
        --gzip \
        --file swift-toolchain.tar.gz \
        --directory /opt/swift \
        --strip-components 1 \
    && /opt/swift/usr/bin/swift --version

ENV PATH="/opt/swift/usr/bin:${PATH}"

COPY .build-context/CTensorRTShim.cpp ./CTensorRTShim.cpp
COPY .build-context/CRG10Preprocessor.cpp ./CRG10Preprocessor.cpp
COPY .build-context/CTensorRTShim.h ./CTensorRTShim.h
COPY Probe.cpp ./Probe.cpp
COPY .build-context/Package.swift ./swift-source/Package.swift
COPY .build-context/Package.resolved ./swift-source/Package.resolved
COPY .build-context/Sources/CTensorRTShim ./swift-source/Sources/CTensorRTShim
COPY .build-context/Sources/OpenVisionTensorRT ./swift-source/Sources/OpenVisionTensorRT
COPY .build-context/Sources/OpenVisionTensorRTRuntimeProbe ./swift-source/Sources/OpenVisionTensorRTRuntimeProbe
COPY .build-context/Sources/OpenVisionTensorRTSwiftProbe ./swift-source/Sources/OpenVisionTensorRTSwiftProbe
COPY .build-context/Tests/OpenVisionTensorRTTests ./swift-source/Tests/OpenVisionTensorRTTests

RUN find /build/swift-source \
        -maxdepth 4 \
        -type f \
        -print \
        | sort

RUN g++ \
        -std=c++17 \
        -O2 \
        -DOVTRT_ENABLE_TEST_HOOKS=1 \
        -Wall \
        -Wextra \
        -Werror \
        -isystem /usr/local/cuda-13.2/include \
        -isystem TensorRT-10.16/include \
        -I . \
        CTensorRTShim.cpp \
        CRG10Preprocessor.cpp \
        Probe.cpp \
        -ldl \
        -o openvision-tensorrt-runtime-probe

RUN g++ \
        -std=c++17 \
        -O1 \
        -g \
        -fno-omit-frame-pointer \
        -DOVTRT_ENABLE_TEST_HOOKS=1 \
        -fsanitize=address,undefined \
        -Wall \
        -Wextra \
        -Werror \
        -isystem /usr/local/cuda-13.2/include \
        -isystem TensorRT-10.16/include \
        -I . \
        CTensorRTShim.cpp \
        CRG10Preprocessor.cpp \
        Probe.cpp \
        -ldl \
        -o openvision-tensorrt-runtime-probe-sanitized

RUN CPLUS_INCLUDE_PATH="/usr/local/cuda-13.2/include:/build/TensorRT-10.16/include" \
    swift build \
        --package-path /build/swift-source \
        --configuration release \
        --product openvision-tensorrt-swift-probe \
        -Xcc -I/usr/local/cuda-13.2/include \
        -Xcc -I/build/TensorRT-10.16/include \
        --static-swift-stdlib

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
COPY --from=builder \
    /build/swift-source/.build/release/openvision-tensorrt-swift-probe \
    /usr/local/bin/openvision-tensorrt-swift-probe
COPY --from=builder \
    /usr/local/cuda-13.2/targets/sbsa-linux/lib/libnvrtc.so* \
    /usr/local/lib/
COPY --from=builder \
    /usr/local/cuda-13.2/targets/sbsa-linux/lib/libnvrtc-builtins.so* \
    /usr/local/lib/

RUN ldconfig

ENTRYPOINT ["/bin/sh", "-c", "ASAN_OPTIONS=detect_leaks=0 /usr/local/bin/openvision-tensorrt-runtime-probe-sanitized >/dev/null && C_RESULT=$(/usr/local/bin/openvision-tensorrt-runtime-probe) && SWIFT_RESULT=$(/usr/local/bin/openvision-tensorrt-swift-probe) && printf '{\"cProbe\":%s,\"swiftProbe\":%s}\\n' \"$C_RESULT\" \"$SWIFT_RESULT\""]
