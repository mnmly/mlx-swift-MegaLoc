import SwiftUI

// Design tokens — one place for spacing, radii, colour and motion so every
// value is deliberate (apple-design §16 "Craft": nothing is random).

enum Theme {
    // 8-pt spacing grid.
    static let s1: CGFloat = 8
    static let s2: CGFloat = 12
    static let s3: CGFloat = 16
    static let s4: CGFloat = 24
    static let s5: CGFloat = 32

    static let panelRadius: CGFloat = 20
    static let cardRadius: CGFloat = 14
    static let controlRadius: CGFloat = 10

    /// Similarity accent ramp: cool (low) → warm (high match).
    static func matchColor(_ similarity: Float) -> Color {
        // Clamp to a sensible retrieval range so mid-scores still read.
        let t = Double(max(0, min(1, (similarity - 0.4) / 0.6)))
        return Color(hue: 0.58 - 0.58 * t, saturation: 0.72, brightness: 0.95)
    }

    // Springs. Critically damped by default (apple-design §4); the grid
    // re-order after a momentum-free tap should settle, not bounce.
    static let settle = Animation.spring(response: 0.42, dampingFraction: 1.0)
    static let arrival = Animation.spring(response: 0.4, dampingFraction: 0.82)

    /// Motion that respects Reduce Motion — a cross-fade stand-in (§14).
    static func motion(_ base: Animation, reduce: Bool) -> Animation {
        reduce ? .easeInOut(duration: 0.2) : base
    }
}

extension Text {
    /// Large numeric readout: tighten tracking as type grows (apple-design §15).
    func bigNumber() -> some View {
        font(.system(size: 34, weight: .semibold, design: .rounded))
            .monospacedDigit()
            .tracking(-0.5)
    }

    /// Small uppercase label above a value.
    func fieldLabel() -> some View {
        font(.system(size: 11, weight: .semibold))
            .tracking(0.6)
            .textCase(.uppercase)
            .foregroundStyle(.secondary)
    }
}
