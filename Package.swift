// swift-tools-version: 6.4

import PackageDescription

let package = Package(
    name: "OpenVisionTensorRT",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "OpenVisionTensorRT",
            targets: ["OpenVisionTensorRT"]
        ),
        .executable(
            name: "openvision-tensorrt-probe",
            targets: ["OpenVisionTensorRTRuntimeProbe"]
        ),
        .executable(
            name: "openvision-tensorrt-swift-probe",
            targets: ["OpenVisionTensorRTSwiftProbe"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/1amageek/OpenVision.git",
            branch: "main"
        )
    ],
    targets: [
        .target(
            name: "CTensorRTShim",
            linkerSettings: [
                .linkedLibrary(
                    "dl",
                    .when(platforms: [.linux])
                )
            ]
        ),
        .target(
            name: "OpenVisionTensorRT",
            dependencies: [
                "CTensorRTShim",
                "OpenVision"
            ]
        ),
        .executableTarget(
            name: "OpenVisionTensorRTRuntimeProbe",
            dependencies: ["OpenVisionTensorRT"]
        ),
        .executableTarget(
            name: "OpenVisionTensorRTSwiftProbe",
            dependencies: [
                "OpenVisionTensorRT",
                "OpenVision"
            ]
        ),
        .testTarget(
            name: "OpenVisionTensorRTTests",
            dependencies: ["OpenVisionTensorRT"]
        )
    ],
    swiftLanguageModes: [.v6],
    cxxLanguageStandard: .cxx17
)
