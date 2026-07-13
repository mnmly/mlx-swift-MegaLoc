import ArgumentParser
import Foundation
import MLXMegaLoc

@main
struct MegaLocCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "megaloc-cli",
        abstract: "MegaLoc visual place recognition (MLX Swift).",
        subcommands: [Download.self, Embed.self, Similarity.self, Rank.self, Bench.self]
    )
}

/// Resolve the weights path: explicit flag → HF cache → helpful error.
func resolveWeights(_ explicit: String?) throws -> URL {
    if let explicit {
        let url = URL(fileURLWithPath: explicit)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Weights not found at \(explicit)")
        }
        return url
    }
    if let cached = MegaLocHub.cachedModelURL() { return cached }
    throw ValidationError(
        "No weights found. Run `megaloc-cli download`, or pass --weights <model.safetensors>.")
}

struct WeightsOption: ParsableArguments {
    @Option(name: .long, help: "Path to model.safetensors (defaults to the HuggingFace cache).")
    var weights: String?
}
