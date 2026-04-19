import Foundation
#if canImport(UIKit)
import UIKit
#endif

enum CTAPIError: LocalizedError {
    case invalidBaseURL
    case invalidResponse
    case requestFailed(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Backend URL is invalid."
        case .invalidResponse:
            return "The backend returned an invalid response."
        case let .requestFailed(code, message):
            return "Request failed (\(code)): \(message)"
        }
    }
}

struct APIClient {
    let baseURL: URL

    init(baseURLString: String) throws {
        guard let url = URL(string: baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw CTAPIError.invalidBaseURL
        }
        baseURL = url
    }

    func resolve(_ path: String) -> URL {
        if let absolute = URL(string: path), absolute.scheme != nil {
            return absolute
        }
        return URL(string: path, relativeTo: baseURL)!.absoluteURL
    }

    func getJSON<T: Decodable>(_ path: String, as type: T.Type = T.self) async throws -> T {
        let (data, response) = try await URLSession.shared.data(from: resolve(path))
        try validate(response: response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    func postJSON<T: Decodable, Body: Encodable>(_ path: String, body: Body, as type: T.Type = T.self) async throws -> T {
        var request = URLRequest(url: resolve(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    func postJSON<T: Decodable>(_ path: String, as type: T.Type = T.self) async throws -> T {
        var request = URLRequest(url: resolve(path))
        request.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    func download(_ path: String, fileName: String? = nil) async throws -> URL {
        let sourceURL = resolve(path)
        let (temporaryURL, response) = try await URLSession.shared.download(from: sourceURL)
        try validate(response: response, data: Data())

        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let destination = caches.appendingPathComponent(fileName ?? sourceURL.lastPathComponent)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporaryURL, to: destination)
        return destination
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw CTAPIError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw CTAPIError.requestFailed(http.statusCode, message)
        }
    }
}

@MainActor
final class CTViewModel: ObservableObject {
    @Published var backendBaseURL: String {
        didSet {
            UserDefaults.standard.set(backendBaseURL, forKey: "backendBaseURL")
        }
    }

    @Published var slices: [SliceRecord] = []
    @Published var assets: [VisionAssetManifest] = []
    @Published var activeAssetID: String = ""
    @Published var selectedSliceIDs: Set<String> = []
    @Published var prompt: String = "Summarize the visible anatomy and call out any concerning findings or uncertainty."
    @Published var analysisText: String = ""
    @Published var statusText: String = "Connect to the CT backend."
    @Published var isBusy = false
    @Published var visibleAnatomyLabels: Set<String> = []
    @Published var modelFileURL: URL?
    @Published var selectedOrganName: String = ""
    @Published var modelRotationX: Float = -0.12
    @Published var modelRotationY: Float = 0
    @Published var modelInteractionScale: Float = 1
    @Published var isModelInteractionLocked = false
    @Published var snapshotQuestion: String = "What should I notice in this 3D view?"
    @Published var snapshotAnalysisText: String = ""
    @Published var snapshotStatusText: String = "Capture the current 3D view before asking."
    @Published var capturedSnapshotBase64: String?

    init() {
        backendBaseURL = UserDefaults.standard.string(forKey: "backendBaseURL") ?? "http://127.0.0.1:8000"
    }

    var activeAsset: VisionAssetManifest? {
        assets.first(where: { $0.id == activeAssetID }) ?? assets.first
    }

    var sortedPresetNames: [String] {
        guard let activeAsset else { return [] }
        let preferred = ["All", "Chest Core", "Heart", "Vessels", "Lungs", "Bones"]
        let remaining = activeAsset.presets.keys.filter { !preferred.contains($0) }.sorted()
        return preferred.filter { activeAsset.presets[$0] != nil } + remaining
    }

    func client() throws -> APIClient {
        try APIClient(baseURLString: backendBaseURL)
    }

    func resolvedURL(_ path: String) -> URL? {
        try? client().resolve(path)
    }

    func refresh() async {
        await runBusy("Refreshing backend data") {
            let api = try client()
            async let sliceResult: [SliceRecord] = api.getJSON("/api/slices?limit=100")
            async let assetResult: [VisionAssetManifest] = api.getJSON("/api/visionos/assets")
            slices = try await sliceResult
            assets = try await assetResult
            if activeAssetID.isEmpty || !assets.contains(where: { $0.id == activeAssetID }) {
                activeAssetID = assets.first?.id ?? ""
            }
            resetVisibleLabelsFromActiveAsset()
            try await downloadActiveModel()
            statusText = assets.isEmpty ? "No Vision Pro asset yet. Export one from the backend." : "Ready."
        }
    }

    func exportBalancedAsset() async {
        await runBusy("Exporting Vision Pro asset") {
            let api = try client()
            let path = "/api/visionos/export?source_id=totalseg_CT_chest_realistic&quality=balanced&force=false"
            let exported: VisionAssetManifest = try await api.postJSON(path)
            if let existingIndex = assets.firstIndex(where: { $0.id == exported.id }) {
                assets[existingIndex] = exported
            } else {
                assets.insert(exported, at: 0)
            }
            activeAssetID = exported.id
            resetVisibleLabelsFromActiveAsset()
            try await downloadActiveModel()
            statusText = "Vision Pro model exported and loaded."
        }
    }

    func setActiveAsset(_ assetID: String) async {
        activeAssetID = assetID
        resetVisibleLabelsFromActiveAsset()
        await runBusy("Loading selected model") {
            try await downloadActiveModel()
            statusText = "Loaded \(assetID)."
        }
    }

    func toggleSlice(_ id: String) {
        if selectedSliceIDs.contains(id) {
            selectedSliceIDs.remove(id)
        } else if selectedSliceIDs.count < 5 {
            selectedSliceIDs.insert(id)
            statusText = "\(selectedSliceIDs.count) slice\(selectedSliceIDs.count == 1 ? "" : "s") selected for inference."
        } else {
            statusText = "Select up to five slices for one inference run."
        }
    }

    func applyPreset(_ name: String) {
        guard let labels = activeAsset?.presets[name] else { return }
        visibleAnatomyLabels = Set(labels)
    }

    func toggleAnatomy(_ label: String) {
        if visibleAnatomyLabels.contains(label) {
            visibleAnatomyLabels.remove(label)
        } else {
            visibleAnatomyLabels.insert(label)
        }
    }

    func setVisibility(labels: [String], visible: Bool) {
        if visible {
            visibleAnatomyLabels.formUnion(labels)
        } else {
            visibleAnatomyLabels.subtract(labels)
        }
    }

    func showOnly(labels: [String]) {
        visibleAnatomyLabels = Set(labels)
    }

    func updateModelViewState(rotationX: Float, rotationY: Float, scale: Float) {
        modelRotationX = rotationX
        modelRotationY = rotationY
        modelInteractionScale = scale
    }

    func captureCurrent3DView() {
        guard let snapshot = currentWindowPNGBase64() else {
            snapshotStatusText = "Could not capture the current Vision window."
            return
        }
        capturedSnapshotBase64 = snapshot
        isModelInteractionLocked = true
        snapshotStatusText = "3D view captured. Ask a question about this frozen view."
    }

    func releaseCurrent3DView() {
        capturedSnapshotBase64 = nil
        snapshotAnalysisText = ""
        isModelInteractionLocked = false
        snapshotStatusText = "Capture the current 3D view before asking."
    }

    func analyzeSelectedSlices() async {
        guard !selectedSliceIDs.isEmpty else {
            statusText = "Select one to five CT slices first."
            return
        }
        await runBusy("Calling Gemini") {
            let request = AnalyzeRequest(
                sliceIDs: Array(selectedSliceIDs).sorted(),
                userNote: prompt,
                dryRun: false,
                provider: "gemini"
            )
            let response: AnalyzeResponse = try await client().postJSON("/api/analyze", body: request)
            analysisText = response.analysis
            statusText = "\(response.provider ?? "gemini") \(response.model) returned \(response.mode)."
        }
    }

    func analyzeCurrent3DView() async {
        if capturedSnapshotBase64 == nil {
            captureCurrent3DView()
        }
        guard let snapshot = capturedSnapshotBase64 else {
            statusText = "Capture the 3D view first."
            return
        }
        isModelInteractionLocked = true
        await runBusy("Calling Gemini on frozen 3D view") {
            let request = SnapshotAnalyzeRequest(
                imageBase64: snapshot,
                userNote: snapshotQuestion,
                assetID: activeAsset?.id,
                visibleLabels: Array(visibleAnatomyLabels).sorted(),
                rotationX: modelRotationX,
                rotationY: modelRotationY,
                scale: modelInteractionScale,
                dryRun: false
            )
            let response: AnalyzeResponse = try await client().postJSON("/api/analyze-3d-snapshot", body: request)
            snapshotAnalysisText = response.analysis
            snapshotStatusText = "\(response.provider ?? "gemini") \(response.model) analyzed the frozen 3D view."
            statusText = snapshotStatusText
        }
        isModelInteractionLocked = capturedSnapshotBase64 != nil
    }

    private func resetVisibleLabelsFromActiveAsset() {
        if let all = activeAsset?.presets["All"] {
            visibleAnatomyLabels = Set(all)
        } else {
            visibleAnatomyLabels = Set(activeAsset?.anatomy.map(\.label) ?? [])
        }
    }

    private func downloadActiveModel() async throws {
        guard let activeAsset else {
            modelFileURL = nil
            return
        }
        let api = try client()
        modelFileURL = try await api.download(activeAsset.usdzURL, fileName: "\(activeAsset.id).usdz")
    }

    private func currentWindowPNGBase64() -> String? {
        #if canImport(UIKit)
        guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first,
              let window = scene.windows.first(where: { $0.isKeyWindow }) ?? scene.windows.first else {
            return nil
        }
        let bounds = window.bounds
        guard bounds.width > 0, bounds.height > 0 else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(bounds: bounds, format: format)
        let image = renderer.image { _ in
            window.drawHierarchy(in: bounds, afterScreenUpdates: true)
        }
        return image.pngData()?.base64EncodedString()
        #else
        return nil
        #endif
    }

    private func runBusy(_ message: String, operation: () async throws -> Void) async {
        isBusy = true
        statusText = message
        do {
            try await operation()
        } catch {
            statusText = error.localizedDescription
        }
        isBusy = false
    }
}
