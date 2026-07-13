import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(RetrievalEngine.self) private var engine
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let columns = [GridItem(.adaptive(minimum: 168, maximum: 220), spacing: Theme.s3)]

    var body: some View {
        ZStack {
            background

            VStack(spacing: 0) {
                toolbar
                content
            }

            if !engine.hasModel {
                ModelGateView()
                    .transition(.opacity.combined(with: .scale(scale: 0.96)))
            }
        }
        .animation(Theme.motion(Theme.settle, reduce: reduceMotion), value: engine.hasModel)
        .onAppear { engine.loadCachedModelIfAvailable() }
        .frame(minWidth: 900, minHeight: 620)
    }

    // MARK: background

    private var background: some View {
        LinearGradient(
            colors: [Color(nsColor: .windowBackgroundColor),
                     Color(nsColor: .underPageBackgroundColor)],
            startPoint: .top, endPoint: .bottom
        )
        .ignoresSafeArea()
    }

    // MARK: toolbar (translucent floating chrome)

    private var toolbar: some View {
        HStack(spacing: Theme.s3) {
            HStack(spacing: Theme.s2) {
                Image(systemName: "mappin.and.ellipse")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text("MegaLoc").font(.system(size: 15, weight: .semibold))
                    Text("Visual place recognition").font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }

            Spacer()

            if engine.isEmbedding {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Embedding…").font(.system(size: 12)).foregroundStyle(.secondary)
                }
                .transition(.opacity)
            }

            if engine.hasModel {
                Button {
                    engine.presentAddImages()
                } label: {
                    Label("Add Images", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
        }
        .padding(.horizontal, Theme.s4)
        .padding(.vertical, Theme.s2)
        .background(.regularMaterial)
        .overlay(alignment: .bottom) {
            Divider().opacity(0.5)
        }
    }

    // MARK: content

    @ViewBuilder
    private var content: some View {
        if engine.database.isEmpty && engine.hasModel {
            emptyState
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.s3) {
                    hint
                    LazyVGrid(columns: columns, spacing: Theme.s3) {
                        ForEach(engine.ranked) { item in
                            ImageCard(item: item)
                                .id(item.id)
                        }
                    }
                    .animation(Theme.motion(Theme.settle, reduce: reduceMotion),
                               value: engine.ranked.map(\.id))
                }
                .padding(Theme.s4)
            }
        }
    }

    private var hint: some View {
        HStack(spacing: Theme.s2) {
            Image(systemName: engine.query == nil ? "hand.tap" : "sparkle.magnifyingglass")
                .foregroundStyle(.secondary)
            Text(engine.query == nil
                 ? "Tap an image to use it as the query — the rest re-rank by place similarity."
                 : "Ranked by cosine similarity to the query. Tap the query again to clear.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var emptyState: some View {
        VStack(spacing: Theme.s3) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 46, weight: .light))
                .foregroundStyle(.secondary)
            Text("No images yet").font(.title3.weight(.semibold))
            Text("Add a few photos of places to build a retrieval database.")
                .font(.system(size: 13)).foregroundStyle(.secondary)
            Button { engine.presentAddImages() } label: {
                Label("Add Images", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(Theme.s5)
    }
}
