import SwiftUI

@main
struct CTVisionDemoApp: App {
    @StateObject private var viewModel = CTViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
        .defaultSize(width: 1680, height: 980)

        ImmersiveSpace(id: "CTVolumeSpace") {
            ImmersiveVolumeView()
                .environmentObject(viewModel)
        }
    }
}
