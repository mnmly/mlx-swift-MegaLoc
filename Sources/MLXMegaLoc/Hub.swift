import Foundation

// Minimal HuggingFace download for the MegaLoc checkpoint. Kept dependency-free
// (URLSession only) and writes into the standard `~/.cache/huggingface` layout
// so it interoperates with the `hf` / `huggingface_hub` CLI and Python code.

public enum MegaLocHub {
    public static let repoId = "gberton/MegaLoc"
    public static let modelFile = "model.safetensors"
    public static let configFile = "config.json"
    public static let revision = "main"

    /// HuggingFace hub cache root, honouring `HF_HOME` / `HUGGINGFACE_HUB_CACHE`,
    /// defaulting to `~/.cache/huggingface/hub`.
    public static var cacheRoot: URL {
        let env = ProcessInfo.processInfo.environment
        if let hub = env["HUGGINGFACE_HUB_CACHE"] { return URL(fileURLWithPath: hub) }
        if let home = env["HF_HOME"] {
            return URL(fileURLWithPath: home).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// The `models--org--name` directory for the MegaLoc repo.
    public static var repoDir: URL {
        cacheRoot.appendingPathComponent("models--gberton--MegaLoc")
    }

    /// Search the cache for an already-downloaded `model.safetensors` (any
    /// snapshot revision, following the standard symlink layout). Returns the
    /// first existing match, or `nil`.
    public static func cachedModelURL() -> URL? {
        let fm = FileManager.default
        let snapshots = repoDir.appendingPathComponent("snapshots")
        guard let revs = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil
        ) else { return nil }
        for rev in revs {
            let candidate = rev.appendingPathComponent(modelFile)
            if fm.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    public enum HubError: LocalizedError {
        case httpStatus(Int)
        case badURL

        public var errorDescription: String? {
            switch self {
            case .httpStatus(let code): return "HuggingFace returned HTTP \(code)"
            case .badURL: return "Invalid HuggingFace URL"
            }
        }
    }

    private static func resolveURL(_ file: String) -> URL? {
        URL(string: "https://huggingface.co/\(repoId)/resolve/\(revision)/\(file)")
    }

    /// Download `model.safetensors` (and `config.json`) into the HF cache and
    /// return the local model URL. If already present, returns it immediately.
    /// `progress` receives a fraction in `0...1` for the (large) model file.
    public static func download(
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        if let existing = cachedModelURL() { progress?(1.0); return existing }

        let fm = FileManager.default
        let snapshotDir = repoDir.appendingPathComponent("snapshots/\(revision)")
        try fm.createDirectory(at: snapshotDir, withIntermediateDirectories: true)

        // Small config first (best-effort).
        if let cfgURL = resolveURL(configFile) {
            if let (data, resp) = try? await URLSession.shared.data(from: cfgURL),
               (resp as? HTTPURLResponse)?.statusCode == 200 {
                try? data.write(to: snapshotDir.appendingPathComponent(configFile))
            }
        }

        guard let modelURL = resolveURL(modelFile) else { throw HubError.badURL }
        let dest = snapshotDir.appendingPathComponent(modelFile)

        let downloader = Downloader(progress: progress)
        let temp = try await downloader.download(from: modelURL)

        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: temp, to: dest)
        return dest
    }

    /// URLSession download-task wrapper that reports progress and hands back the
    /// completed temp file. Efficient (kernel-level streaming), unlike the
    /// byte-by-byte `AsyncBytes` API.
    private final class Downloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        private let progress: (@Sendable (Double) -> Void)?
        private var continuation: CheckedContinuation<URL, Error>?
        private var stagedURL: URL?

        init(progress: (@Sendable (Double) -> Void)?) { self.progress = progress }

        func download(from url: URL) async throws -> URL {
            try await withCheckedThrowingContinuation { cont in
                self.continuation = cont
                let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
                session.downloadTask(with: url).resume()
                session.finishTasksAndInvalidate()
            }
        }

        func urlSession(
            _ session: URLSession, downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            guard totalBytesExpectedToWrite > 0 else { return }
            progress?(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
        }

        func urlSession(
            _ session: URLSession, downloadTask: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {
            // The temp file is deleted when this delegate returns; move it aside.
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                continuation?.resume(throwing: HubError.httpStatus(http.statusCode))
                continuation = nil
                return
            }
            let staged = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".safetensors")
            do {
                try FileManager.default.moveItem(at: location, to: staged)
                stagedURL = staged
            } catch {
                continuation?.resume(throwing: error)
                continuation = nil
            }
        }

        func urlSession(
            _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
        ) {
            if let error {
                continuation?.resume(throwing: error)
            } else if let staged = stagedURL {
                progress?(1.0)
                continuation?.resume(returning: staged)
            } else {
                continuation?.resume(throwing: HubError.badURL)
            }
            continuation = nil
        }
    }
}
