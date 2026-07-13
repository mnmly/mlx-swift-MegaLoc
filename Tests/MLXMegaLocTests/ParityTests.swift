import Foundation
import MLX
import XCTest

@testable import MLXMegaLoc

/// Numerical-parity tests against fixtures generated from the reference PyTorch
/// model (see Tools/generate_fixtures.py).
///
/// The reference was run on CPU (torch default). MLX on the **CPU** device
/// reproduces it to ~1e-4 (a true correctness proof); MLX on the **GPU** (Metal)
/// differs only by fp32 matmul accumulation order, which is inherent and leaves
/// the descriptor cosine at ~0.9999 — indistinguishable for retrieval.
///
/// Weights (914 MB) are not bundled; point `MEGALOC_WEIGHTS` at a local
/// `model.safetensors`, or rely on the default HuggingFace cache path.
final class ParityTests: XCTestCase {

    static let defaultWeights =
        "\(NSHomeDirectory())/.cache/huggingface/hub/models--gberton--MegaLoc"
        + "/snapshots/7cb9f7970d366fdf059963d04d372e503e8e9df9/model.safetensors"

    private func weightsURL() throws -> URL {
        let path = ProcessInfo.processInfo.environment["MEGALOC_WEIGHTS"] ?? Self.defaultWeights
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("MegaLoc weights not found at \(path). Set MEGALOC_WEIGHTS.")
        }
        return URL(fileURLWithPath: path)
    }

    private func fixtureURL(_ name: String) throws -> URL {
        guard let url = Bundle.module.url(
            forResource: name, withExtension: "safetensors", subdirectory: "Fixtures")
        else { throw XCTSkip("Fixture \(name) not found in bundle.") }
        return url
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32).reshaped([-1]) - b.asType(.float32).reshaped([-1]))
            .max().item(Float.self)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        (a.reshaped([-1]) * b.reshaped([-1])).sum().item(Float.self)
    }

    /// Runs the staged comparison on the current default device.
    private func runStages(
        _ fx: [String: MLXArray], _ model: MegaLoc,
        patchTol: Float, saladTol: Float, finalTol: Float, cosTol: Float, label: String
    ) {
        let input = fx["input_nchw"]!.transposed(axes: [0, 2, 3, 1])   // NCHW -> NHWC

        let (patches, cls) = model.backbone(input)
        let refPatches = fx["patch_features_nchw"]!.transposed(axes: [0, 2, 3, 1])
        eval(patches, cls)
        let dPatch = maxAbs(patches, refPatches)
        let dCls = maxAbs(cls, fx["cls_token"]!)

        let salad = model.aggregator.agg(patches: patches, cls: cls)
        eval(salad)
        let dSalad = maxAbs(salad, fx["salad"]!)

        let linearOut = model.aggregator.linear(salad)
        eval(linearOut)
        let dLin = maxAbs(linearOut, fx["linear_out"]!)

        let final = model(input)
        eval(final)
        let dFinal = maxAbs(final, fx["final"]!)
        let cos = cosine(final, fx["final"]!)

        print("[\(label)] patch=\(fmt(dPatch)) cls=\(fmt(dCls)) salad=\(fmt(dSalad)) "
            + "linear=\(fmt(dLin)) final=\(fmt(dFinal)) cos=\(String(format: "%.6f", cos))")

        XCTAssertLessThan(dPatch, patchTol, "[\(label)] patch features")
        XCTAssertLessThan(dCls, patchTol, "[\(label)] cls token")
        XCTAssertLessThan(dSalad, saladTol, "[\(label)] salad")
        XCTAssertLessThan(dLin, finalTol, "[\(label)] linear head")
        XCTAssertLessThan(dFinal, finalTol, "[\(label)] final descriptor")
        XCTAssertGreaterThan(cos, cosTol, "[\(label)] descriptor cosine")
    }

    private func fmt(_ x: Float) -> String { String(format: "%.2e", x) }

    /// The CPU proof runs the full fp32 backbone on the CPU device — a rigorous
    /// correctness check, but slow. It is opt-in so the default `xcodebuild test`
    /// stays fast (the GPU test is the routine check). Enable it by either:
    ///   - `touch ~/.megaloc-run-cpu-parity` (reliable for the `xcodebuild test`
    ///     CLI, which does not forward shell env to the test runner), or
    ///   - setting `MEGALOC_CPU_PARITY=1` (honored where env *is* forwarded — an
    ///     Xcode scheme's test environment, or `swift test`).
    static var cpuParityEnabled: Bool {
        if ProcessInfo.processInfo.environment["MEGALOC_CPU_PARITY"] == "1" { return true }
        let marker = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".megaloc-run-cpu-parity")
        return FileManager.default.fileExists(atPath: marker.path)
    }

    /// CPU device: a strict correctness proof (matches PyTorch CPU to ~1e-4).
    /// The looser patch tolerance covers DINOv2's few high-norm artifact tokens.
    /// Opt-in — see ``cpuParityEnabled``.
    func testParity518_CPU() throws {
        try XCTSkipUnless(
            Self.cpuParityEnabled,
            "CPU parity proof is opt-in (slow). Enable with `touch ~/.megaloc-run-cpu-parity`"
            + " or MEGALOC_CPU_PARITY=1.")
        let fx = try MLX.loadArrays(url: try fixtureURL("parity_518"))
        let model = try MegaLoc.load(weights: try weightsURL(), dtype: .float32)
        Device.withDefaultDevice(Device(.cpu)) {
            runStages(fx, model, patchTol: 3e-3, saladTol: 5e-5, finalTol: 5e-5,
                      cosTol: 0.999999, label: "CPU")
        }
    }

    /// GPU (Metal) production path: realistic fp32 tolerance; the retrieval
    /// descriptor is effectively identical (cosine > 0.9999).
    func testParity518_GPU() throws {
        let fx = try MLX.loadArrays(url: try fixtureURL("parity_518"))
        let model = try MegaLoc.load(weights: try weightsURL(), dtype: .float32)
        runStages(fx, model, patchTol: 2.0, saladTol: 2e-3, finalTol: 3e-3,
                  cosTol: 0.9998, label: "GPU")
    }
}
