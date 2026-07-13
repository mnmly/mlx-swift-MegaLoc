import SwiftUI

/// Modal gate shown until a checkpoint is loaded. Offers three paths: load the
/// copy already in ~/.cache/huggingface, download it there, or pick a file.
struct ModelGateView: View {
    @Environment(RetrievalEngine.self) private var engine
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            // Dimming scrim pushes the background back (apple-design §12).
            Color.black.opacity(0.28).ignoresSafeArea()

            card
                .frame(width: 420)
                .padding(Theme.s5)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                        .strokeBorder(.white.opacity(0.12))
                )
                .shadow(color: .black.opacity(0.3), radius: 30, y: 12)
        }
    }

    @ViewBuilder
    private var card: some View {
        VStack(spacing: Theme.s3) {
            Image(systemName: "mappin.and.ellipse")
                .font(.system(size: 40, weight: .semibold))
                .foregroundStyle(.tint)
                .padding(.bottom, 2)

            Text("Load MegaLoc")
                .font(.system(size: 20, weight: .semibold))
            Text("A ~914 MB DINOv2 + SALAD checkpoint from HuggingFace, cached in ~/.cache/huggingface.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)

            switch engine.modelState {
            case .downloading(let frac): downloadingView(frac)
            case .loading: loadingView
            default: actions
            }

            if case .failed(let message) = engine.modelState {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var actions: some View {
        VStack(spacing: Theme.s2) {
            if let cached = engine.cachedModelURL {
                Button {
                    engine.loadModel(at: cached)
                } label: {
                    Label("Load cached model", systemImage: "checkmark.seal.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                Text(cached.path)
                    .font(.system(size: 10)).foregroundStyle(.tertiary)
                    .lineLimit(1).truncationMode(.middle)
            } else {
                Button {
                    engine.download()
                } label: {
                    Label("Download model", systemImage: "arrow.down.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }

            Button {
                engine.presentChooseModel()
            } label: {
                Label("Choose file…", systemImage: "folder")
                    .frame(maxWidth: .infinity)
            }
            .controlSize(.large)
        }
        .padding(.top, Theme.s1)
    }

    private func downloadingView(_ frac: Double) -> some View {
        VStack(spacing: Theme.s2) {
            ProgressView(value: frac) {
                Text("Downloading…").font(.system(size: 12)).foregroundStyle(.secondary)
            } currentValueLabel: {
                Text("\(Int(frac * 100))%").font(.system(size: 12).monospacedDigit())
            }
            .progressViewStyle(.linear)
        }
        .padding(.top, Theme.s2)
    }

    private var loadingView: some View {
        HStack(spacing: Theme.s2) {
            ProgressView().controlSize(.small)
            Text("Loading weights…").font(.system(size: 13)).foregroundStyle(.secondary)
        }
        .padding(.top, Theme.s2)
    }
}
