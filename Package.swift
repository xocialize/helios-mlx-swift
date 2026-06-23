// swift-tools-version: 6.2
// helios-mlx-swift — Swift/MLX port of Helios-Distilled (PKU-YuanGroup), a 14B autoregressive
// minute-scale text-to-video model, on the `wan-core` substrate. Second `wan-core` consumer after
// Bernini-R. Python oracle: mlx-video PR #21 (dmunch/mlx-video @ helios, pinned 27902e7). The
// backbone is 100% wan-core reuse; the net-new Swift is the AR history/memory delta + DMD scheduler.
// See PORTING-SPEC.md. The MLXToolKit engine wrapper (MLXHelios) is added at S7.

import PackageDescription

let package = Package(
    name: "Helios",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Helios", targets: ["Helios"]),
        // The MLXEngine ModelPackage wrapper (S7) — register with MLXServeEngine.
        .library(name: "MLXHelios", targets: ["MLXHelios"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // The neutral Wan substrate (DiT + VAE + umT5 + RoPE + schedulers + loader). Local path
        // during B1; tagged dep later. Helios reuses it as-is and adds only the AR delta.
        .package(path: "../wan-core-mlx-swift"),
        // MLXEngine contract (MLXToolKit) for the wrapper target.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.9.1"),
    ],
    targets: [
        .target(
            name: "Helios",
            dependencies: [
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Helios"
        ),
        .target(
            name: "MLXHelios",
            dependencies: [
                "Helios",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXHelios"
        ),
        .executableTarget(
            name: "RunHelios",
            dependencies: [
                "Helios",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/RunHelios"
        ),
        .testTarget(
            name: "HeliosTests",
            dependencies: [
                "Helios",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
            ],
            path: "Tests/HeliosTests"
        ),
    ]
)
