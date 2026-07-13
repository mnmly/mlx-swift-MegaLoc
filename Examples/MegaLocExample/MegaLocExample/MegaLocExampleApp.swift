import SwiftUI

@main
struct MegaLocExampleApp: App {
    @State private var engine = RetrievalEngine()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(engine)
                .frame(minWidth: 900, minHeight: 620)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
    }
}
