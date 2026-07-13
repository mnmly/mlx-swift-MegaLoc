import Foundation
import MLX
import XCTest

@testable import MLXMegaLoc

/// Per-block backbone bisection. Guarded by `MEGALOC_STAGES` pointing at a
/// stages safetensors produced by Tools/dump_stages.py (58 MB, not committed).
final class StageBisectTests: XCTestCase {

    func testBackboneStages() throws {
        let stagesPath = ProcessInfo.processInfo.environment["MEGALOC_STAGES"]
            ?? "/tmp/megaloc_stages_518.safetensors"
        guard FileManager.default.fileExists(atPath: stagesPath) else {
            throw XCTSkip("stages file not found at \(stagesPath); run Tools/dump_stages.py")
        }
        let weights = ProcessInfo.processInfo.environment["MEGALOC_WEIGHTS"] ?? ParityTests.defaultWeights
        guard FileManager.default.fileExists(atPath: weights) else {
            throw XCTSkip("weights missing")
        }
        guard let fixURL = Bundle.module.url(
            forResource: "parity_518", withExtension: "safetensors", subdirectory: "Fixtures")
        else { throw XCTSkip("fixture missing") }

        let fx = try MLX.loadArrays(url: fixURL)
        let stages = try MLX.loadArrays(url: URL(fileURLWithPath: stagesPath))
        let model = try MegaLoc.load(weights: URL(fileURLWithPath: weights), dtype: .float32)
        let input = fx["input_nchw"]!.transposed(axes: [0, 2, 3, 1])

        var order = ["emb"]
        for i in 0 ..< 12 { order.append("block_\(i)") }
        order.append("post_norm")

        func report(_ label: String, _ swift: [String: MLXArray]) {
            print("--- \(label) ---")
            for key in order {
                guard let ref = stages[key], let got = swift[key] else { continue }
                eval(got)
                let d = MLX.abs(got.asType(.float32).reshaped([-1]) - ref.asType(.float32).reshaped([-1]))
                print(String(format: "%-10@ maxAbs=%.3e meanAbs=%.3e",
                             key as NSString, d.max().item(Float.self), d.mean().item(Float.self)))
            }
        }

        report("GPU", model.backbone.debugStages(input))
        Device.withDefaultDevice(Device(.cpu)) {
            report("CPU", model.backbone.debugStages(input))
        }
    }
}
