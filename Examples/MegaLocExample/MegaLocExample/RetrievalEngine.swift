import AppKit
import CoreGraphics
import Foundation
import ImageIO
import MLXMegaLoc
import Observation
import UniformTypeIdentifiers

// Frontend view-model. It owns ONLY presentation concerns: the loaded model
// gate, the image database, the current query, and MainActor hops. All compute
// (preprocess, forward, descriptor extraction, ranking) goes through the
// library's `MegaLocSession` — the same shared driver `megaloc-cli` consumes —
// so the GUI can never drift from the CLI (swift-cli-gui-shared-driver).

/// A database image: a display thumbnail plus its (lazily computed) descriptor.
@Observable
final class DBImage: Identifiable {
    let id = UUID()
    let url: URL
    let image: CGImage
    var descriptor: MegaLocDescriptor?
    /// Similarity to the current query (filled when a query is set).
    var similarity: Float?

    init(url: URL, image: CGImage) {
        self.url = url
        self.image = image
    }

    var name: String { url.lastPathComponent }
}

/// CGImage is an immutable, thread-safe CoreFoundation type; box it to cross the
/// task boundary without the compiler flagging non-Sendable capture.
private struct SendableURLs: @unchecked Sendable { let urls: [URL] }

@MainActor
@Observable
final class RetrievalEngine {

    enum ModelState: Equatable {
        case needsModel
        case downloading(Double)      // 0…1
        case loading
        case ready
        case failed(String)
    }

    private(set) var modelState: ModelState = .needsModel
    private(set) var database: [DBImage] = []
    private(set) var queryID: DBImage.ID?
    private(set) var isEmbedding = false

    private var session: MegaLocSession?

    /// Cached checkpoint in ~/.cache/huggingface, if present.
    var cachedModelURL: URL? { MegaLocHub.cachedModelURL() }
    var hasModel: Bool { session != nil }

    var query: DBImage? { database.first { $0.id == queryID } }

    /// Database sorted best-match-first when a query is active, else insertion order.
    var ranked: [DBImage] {
        guard queryID != nil else { return database }
        return database.sorted { a, b in
            if a.id == queryID { return true }
            if b.id == queryID { return false }
            return (a.similarity ?? -1) > (b.similarity ?? -1)
        }
    }

    // MARK: - Model acquisition

    func loadCachedModelIfAvailable() {
        guard session == nil else { return }
        if let url = cachedModelURL { loadModel(at: url) }
    }

    func download() {
        guard case .needsModel = modelState else { return }
        modelState = .downloading(0)
        Task {
            do {
                let url = try await MegaLocHub.download { [weak self] frac in
                    Task { @MainActor in
                        guard let self, case .downloading = self.modelState else { return }
                        self.modelState = .downloading(frac)
                    }
                }
                loadModel(at: url)
            } catch {
                modelState = .failed("Download failed: \(error.localizedDescription)")
            }
        }
    }

    func loadModel(at url: URL) {
        modelState = .loading
        let box = SendableURLs(urls: [url])
        Task {
            do {
                let session = try await Task.detached(priority: .userInitiated) {
                    try MegaLocSession.load(weights: box.urls[0])
                }.value
                self.session = session
                self.modelState = .ready
                self.loadBundledSamples()
            } catch {
                self.modelState = .failed("Couldn't load weights: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Database

    private static let sampleNames = [
        "scene0_view0", "scene0_view1", "scene1_view0",
        "scene1_view1", "scene2_view0", "scene2_view1", "query_scene0",
    ]

    func loadBundledSamples() {
        guard database.isEmpty else { return }
        let urls = Self.sampleNames.compactMap {
            Bundle.main.url(forResource: $0, withExtension: "png")
        }
        addImages(urls: urls)
    }

    func addImages(urls: [URL]) {
        let fresh = urls.compactMap { url -> DBImage? in
            guard let img = MegaLocPreprocess.loadCGImage(url) else { return nil }
            return DBImage(url: url, image: img)
        }
        guard !fresh.isEmpty else { return }
        database.append(contentsOf: fresh)
        embed(fresh)
    }

    func removeImage(_ image: DBImage) {
        database.removeAll { $0.id == image.id }
        if queryID == image.id { queryID = nil }
        recomputeSimilarities()
    }

    func setQuery(_ image: DBImage) {
        queryID = (queryID == image.id) ? nil : image.id
        recomputeSimilarities()
    }

    // MARK: - Compute (off the main actor)

    private func embed(_ images: [DBImage]) {
        guard let session else { return }
        isEmbedding = true
        let box = SendableURLs(urls: images.map { $0.url })
        Task {
            let pairs = await Task.detached(priority: .userInitiated) {
                session.embed(urls: box.urls)
            }.value
            // Match results back to the DBImage entries by URL.
            var byURL: [URL: MegaLocDescriptor] = [:]
            for p in pairs { byURL[p.url] = p.descriptor }
            for img in images { img.descriptor = byURL[img.url] }
            self.isEmbedding = false
            self.recomputeSimilarities()
        }
    }

    /// Cheap post-processing (no model): re-rank the cached descriptors. A GUI
    /// interaction re-runs only this, never the network (shared-driver rule).
    private func recomputeSimilarities() {
        guard let q = query?.descriptor else {
            for img in database { img.similarity = nil }
            return
        }
        for img in database {
            img.similarity = img.descriptor.map { MegaLocSession.similarity(q, $0) }
        }
    }

    // MARK: - Panels

    /// Open panel for image files.
    func presentAddImages() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.png, .jpeg, .heic, .tiff, .image]
        panel.prompt = "Add"
        panel.message = "Add images to the retrieval database"
        if panel.runModal() == .OK { addImages(urls: panel.urls) }
    }

    /// Open panel for the model, starting in the HuggingFace cache directory.
    func presentChooseModel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.init(filenameExtension: "safetensors") ?? .data]
        panel.prompt = "Load"
        panel.message = "Choose model.safetensors"
        panel.directoryURL = MegaLocHub.repoDir
        if panel.runModal() == .OK, let url = panel.url { loadModel(at: url) }
    }
}
