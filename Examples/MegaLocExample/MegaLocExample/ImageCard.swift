import SwiftUI

/// One database image: thumbnail + name, with a similarity readout when a query
/// is active, and a query badge when it is the query.
struct ImageCard: View {
    @Environment(RetrievalEngine.self) private var engine
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let item: DBImage

    @State private var hovering = false

    private var isQuery: Bool { engine.query?.id == item.id }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.s1) {
            thumbnail
            footer
        }
        .padding(Theme.s1)
        .background(cardBackground)
        .overlay(queryRing)
        .scaleEffect(hovering && !isQuery ? 1.02 : 1.0)
        .animation(Theme.motion(Theme.arrival, reduce: reduceMotion), value: hovering)
        .onHover { hovering = $0 }
        .onTapGesture { engine.setQuery(item) }
        .contextMenu {
            Button("Use as Query") { engine.setQuery(item) }
            Button("Remove", role: .destructive) { engine.removeImage(item) }
        }
        .help(item.name)
    }

    private var thumbnail: some View {
        Image(decorative: item.image, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(height: 150)
            .frame(maxWidth: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius - 4, style: .continuous))
            .overlay(alignment: .topTrailing) {
                if isQuery {
                    Label("Query", systemImage: "scope")
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.tint, in: Capsule())
                        .foregroundStyle(.white)
                        .padding(6)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .overlay(alignment: .bottom) { similarityBar }
    }

    @ViewBuilder
    private var similarityBar: some View {
        if !isQuery, let sim = item.similarity {
            GeometryReader { geo in
                let frac = CGFloat(max(0, min(1, (sim - 0.4) / 0.6)))
                ZStack(alignment: .leading) {
                    Capsule().fill(.black.opacity(0.25))
                    Capsule()
                        .fill(Theme.matchColor(sim))
                        .frame(width: max(6, geo.size.width * frac))
                }
                .frame(height: 5)
            }
            .frame(height: 5)
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
    }

    private var footer: some View {
        HStack {
            Text(item.name)
                .font(.system(size: 11, weight: .medium))
                .lineLimit(1).truncationMode(.middle)
                .foregroundStyle(.secondary)
            Spacer()
            if isQuery {
                Text("—").font(.system(size: 12, weight: .semibold).monospacedDigit())
                    .foregroundStyle(.secondary)
            } else if let sim = item.similarity {
                Text(String(format: "%.3f", sim))
                    .font(.system(size: 12, weight: .semibold).monospacedDigit())
                    .foregroundStyle(Theme.matchColor(sim))
            } else if engine.isEmbedding {
                ProgressView().controlSize(.mini)
            }
        }
        .padding(.horizontal, 4)
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
            .fill(.regularMaterial)
            .shadow(color: .black.opacity(hovering ? 0.18 : 0.08),
                    radius: hovering ? 10 : 4, y: hovering ? 4 : 2)
    }

    @ViewBuilder
    private var queryRing: some View {
        if isQuery {
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .strokeBorder(.tint, lineWidth: 2.5)
        }
    }
}
