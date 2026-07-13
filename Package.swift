// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "mlx-swift-MegaLoc",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MLXMegaLoc", targets: ["MLXMegaLoc"]),
        .executable(name: "megaloc-cli", targets: ["megaloc-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "MLXMegaLoc",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXMegaLoc"
        ),
        .executableTarget(
            name: "megaloc-cli",
            dependencies: [
                "MLXMegaLoc",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/megaloc-cli"
        ),
        .testTarget(
            name: "MLXMegaLocTests",
            dependencies: ["MLXMegaLoc"],
            path: "Tests/MLXMegaLocTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation, so normal
// builds and downstream consumers don't have to resolve an extra dependency.
if Context.environment["SPI_GENERATE_DOCS"] == "1"      // Swift Package Index
    || Context.environment["BUILD_DOC"] == "1"          // local / CI
{
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
