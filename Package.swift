// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ConstrainedDecoding",
    // Only platforms the CI workflow builds: iOS (Simulator, via xcodebuild) and macOS.
    // Linux is built and tested too (core module only); it needs no declaration.
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ConstrainedDecoding", targets: ["ConstrainedDecoding"]),
        .library(name: "ConstrainedDecodingUI", targets: ["ConstrainedDecodingUI"]),
    ],
    targets: [
        .target(name: "ConstrainedDecoding"),
        .target(name: "ConstrainedDecodingUI", dependencies: ["ConstrainedDecoding"]),
        .testTarget(name: "ConstrainedDecodingTests", dependencies: ["ConstrainedDecoding"]),
    ]
)
