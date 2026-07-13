import ArgumentParser
import Foundation
import MLX
import MLXMegaLoc

// MARK: - download

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download the MegaLoc checkpoint into ~/.cache/huggingface.")

    func run() async throws {
        if let existing = MegaLocHub.cachedModelURL() {
            print("Already cached: \(existing.path)")
            return
        }
        print("Downloading \(MegaLocHub.repoId)/\(MegaLocHub.modelFile) …")
        let url = try await MegaLocHub.download { frac in
            let pct = Int(frac * 100)
            FileHandle.standardError.write(Data("\rProgress: \(pct)%   ".utf8))
        }
        FileHandle.standardError.write(Data("\n".utf8))
        print("Saved: \(url.path)")
    }
}

// MARK: - embed

struct Embed: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Embed an image and print its descriptor (or save as JSON).")

    @OptionGroup var weightsOpt: WeightsOption
    @Argument(help: "Image file to embed.") var image: String
    @Option(name: .long, help: "Write the descriptor to this JSON file instead of stdout.")
    var out: String?

    func run() throws {
        let weights = try resolveWeights(weightsOpt.weights)
        let session = try MegaLocSession.load(weights: weights)
        guard let img = MegaLocPreprocess.loadCGImage(URL(fileURLWithPath: image)) else {
            throw ValidationError("Could not decode image: \(image)")
        }
        let d = session.embed(image: img)
        if let out {
            let data = try JSONEncoder().encode(d)
            try data.write(to: URL(fileURLWithPath: out))
            print("Wrote \(d.dimension)-dim descriptor to \(out)")
        } else {
            let preview = d.values.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", ")
            print("descriptor dim=\(d.dimension)  [\(preview), …]")
        }
    }
}

// MARK: - similarity

struct Similarity: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Cosine similarity between two images (1.0 = identical place).")

    @OptionGroup var weightsOpt: WeightsOption
    @Argument(help: "First image.") var imageA: String
    @Argument(help: "Second image.") var imageB: String

    func run() throws {
        let weights = try resolveWeights(weightsOpt.weights)
        let session = try MegaLocSession.load(weights: weights)
        guard let a = MegaLocPreprocess.loadCGImage(URL(fileURLWithPath: imageA)),
              let b = MegaLocPreprocess.loadCGImage(URL(fileURLWithPath: imageB)) else {
            throw ValidationError("Could not decode one of the images.")
        }
        let ds = session.embed(images: [a, b])
        let sim = MegaLocSession.similarity(ds[0], ds[1])
        print(String(format: "cosine similarity: %.4f", sim))
    }
}

// MARK: - rank

struct Rank: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Rank a database of images by similarity to a query.")

    @OptionGroup var weightsOpt: WeightsOption
    @Option(name: .long, help: "Query image.") var query: String
    @Argument(help: "Database images.") var database: [String]
    @Option(name: .long, help: "Show only the top K matches.") var topK: Int?

    func run() throws {
        let weights = try resolveWeights(weightsOpt.weights)
        let session = try MegaLocSession.load(weights: weights)
        guard let q = MegaLocPreprocess.loadCGImage(URL(fileURLWithPath: query)) else {
            throw ValidationError("Could not decode query image.")
        }
        let dbURLs = database.map { URL(fileURLWithPath: $0) }
        let dbEmbedded = session.embed(urls: dbURLs)
        let queryDesc = session.embed(image: q)
        let matches = MegaLocSession.rank(
            query: queryDesc, database: dbEmbedded.map { $0.descriptor }, topK: topK)
        print("Ranked \(dbEmbedded.count) images vs query:")
        for (rank, m) in matches.enumerated() {
            let name = dbEmbedded[m.index].url.lastPathComponent
            print(String(format: "  %2d. %.4f  %@", rank + 1, m.similarity, name as NSString))
        }
    }
}

// MARK: - bench

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark the forward pass and check for memory leaks.")

    @OptionGroup var weightsOpt: WeightsOption
    @Option(name: .long, help: "Iterations to time.") var iterations: Int = 50
    @Option(name: .long, help: "Warmup iterations.") var warmup: Int = 5
    @Option(name: .long, help: "Square input size (multiple of 14).") var size: Int = MegaLocPreprocess.defaultImageSize

    func run() throws {
        let weights = try resolveWeights(weightsOpt.weights)
        let session = try MegaLocSession.load(weights: weights, imageSize: size)

        // Deterministic synthetic input so bench needs no image files.
        let input = MLXArray.zeros([1, size, size, 3])

        for _ in 0 ..< warmup { _ = session.embed(batch: input) }

        let start = Memory.snapshot()
        var times: [Double] = []
        times.reserveCapacity(iterations)
        for i in 0 ..< iterations {
            let t0 = Date()
            _ = session.embed(batch: input)
            times.append(Date().timeIntervalSince(t0) * 1000.0)
            if i == iterations / 2 {
                let mid = Memory.snapshot()
                print(String(format: "  [mid] active=%@ cache=%@",
                             human(mid.activeMemory), human(mid.cacheMemory)))
            }
        }
        let end = Memory.snapshot()

        times.sort()
        let mean = times.reduce(0, +) / Double(times.count)
        let p50 = times[times.count / 2]
        let p90 = times[Int(Double(times.count) * 0.9)]
        print(String(format: "\nMegaLoc forward @ %dx%d, %d iters:", size, size, iterations))
        print(String(format: "  mean=%.1f ms  p50=%.1f ms  p90=%.1f ms  (%.1f img/s)",
                     mean, p50, p90, 1000.0 / mean))
        print("  memory: active \(human(start.activeMemory)) → \(human(end.activeMemory)) "
            + "(Δ \(human(end.activeMemory - start.activeMemory))), "
            + "peak \(human(end.peakMemory))")
        let leaked = end.activeMemory - start.activeMemory
        if leaked > 8 * 1024 * 1024 {
            print("  ⚠️  active memory grew by \(human(leaked)) across the loop — investigate a leak.")
        } else {
            print("  ✓ active memory flat across the loop (no leak; large peak is MLX's buffer cache).")
        }
    }

    private func human(_ bytes: Int) -> String {
        let mb = Double(bytes) / (1024 * 1024)
        if abs(mb) >= 1024 { return String(format: "%.2f GB", mb / 1024) }
        return String(format: "%.1f MB", mb)
    }
}
