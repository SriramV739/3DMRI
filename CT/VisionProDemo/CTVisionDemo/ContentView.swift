import Foundation
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var viewModel: CTViewModel
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @State private var immersiveOpen = false
    @State private var modelResetToken = 0

    var body: some View {
        HStack(spacing: 0) {
            ControlPanel(
                immersiveOpen: $immersiveOpen,
                openImmersive: openImmersive,
                closeImmersive: closeImmersive
            )
            .frame(width: 430)

            Divider()

            ModelStage(resetToken: modelResetToken) {
                modelResetToken += 1
            }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            SliceAnalysisSection()
                .frame(width: 470)
        }
        .frame(minWidth: 1540, minHeight: 920)
        .background(.black.opacity(0.08))
        .task {
            if viewModel.assets.isEmpty && viewModel.slices.isEmpty {
                await viewModel.refresh()
            }
        }
    }

    private func openImmersive() {
        Task {
            let result = await openImmersiveSpace(id: "CTVolumeSpace")
            if case .opened = result {
                immersiveOpen = true
            }
        }
    }

    private func closeImmersive() {
        Task {
            await dismissImmersiveSpace()
            immersiveOpen = false
        }
    }
}

private struct ModelStage: View {
    @EnvironmentObject private var viewModel: CTViewModel
    let resetToken: Int
    let resetModel: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                StatusBanner()
                Spacer()
                if let asset = viewModel.activeAsset {
                    Text(asset.id)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Button {
                    resetModel()
                } label: {
                    Label("Reset View", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            RealityModelView(immersive: false, resetToken: resetToken)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipped()
                .contentShape(Rectangle())

            HStack {
                Label(viewModel.selectedOrganName.isEmpty ? "Drag to rotate. Pinch to zoom. Tap anatomy to inspect." : viewModel.selectedOrganName, systemImage: "hand.draw")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
    }
}

private struct StatusBanner: View {
    @EnvironmentObject private var viewModel: CTViewModel

    var body: some View {
        HStack(spacing: 10) {
            if viewModel.isBusy {
                ProgressView()
            }
            Text(viewModel.statusText)
                .lineLimit(2)
            Spacer()
        }
        .font(.caption.weight(.medium))
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
        .frame(maxWidth: 440)
    }
}

private struct ControlPanel: View {
    @EnvironmentObject private var viewModel: CTViewModel
    @Binding var immersiveOpen: Bool
    let openImmersive: () -> Void
    let closeImmersive: () -> Void
    @State private var selectedSection: ControlPanelSection = .organs

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HeaderSummary()

            Picker("Controls", selection: $selectedSection) {
                ForEach(ControlPanelSection.allCases) { section in
                    Text(section.title)
                        .tag(section)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .font(.caption.weight(.semibold))

            Group {
                switch selectedSection {
                case .connection:
                    BackendSection()
                case .model:
                    AssetSection(
                        immersiveOpen: $immersiveOpen,
                        openImmersive: openImmersive,
                        closeImmersive: closeImmersive
                    )
                case .organs:
                    OrganSection()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 16)
        .background(.regularMaterial)
    }
}

private enum ControlPanelSection: String, CaseIterable, Identifiable {
    case connection
    case model
    case organs

    var id: String { rawValue }

    var title: String {
        switch self {
        case .connection: "Backend"
        case .model: "Model"
        case .organs: "Organs"
        }
    }

    var systemImage: String {
        switch self {
        case .connection: "server.rack"
        case .model: "cube.transparent"
        case .organs: "checklist"
        }
    }
}

private struct HeaderSummary: View {
    @EnvironmentObject private var viewModel: CTViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("CT Vision")
                    .font(.title3.weight(.semibold))
                Spacer()
                if viewModel.isBusy {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            HStack(spacing: 8) {
                MetricPill(value: "\(viewModel.slices.count)", label: "slices")
                MetricPill(value: "\(viewModel.assets.count)", label: "assets")
                MetricPill(value: "\(viewModel.visibleAnatomyLabels.count)", label: "shown")
            }
        }
    }
}

private struct MetricPill: View {
    let value: String
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Text(value)
                .fontWeight(.semibold)
            Text(label)
                .foregroundStyle(.secondary)
        }
        .font(.caption)
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct PanelSection<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: systemImage)
                .font(.headline)

            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct BackendSection: View {
    @EnvironmentObject private var viewModel: CTViewModel

    var body: some View {
        PanelSection(title: "Backend", systemImage: "server.rack") {
            TextField("http://127.0.0.1:8000", text: $viewModel.backendBaseURL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.system(.caption, design: .monospaced))
                .textFieldStyle(.roundedBorder)

            HStack(spacing: 10) {
                Button {
                    Task { await viewModel.refresh() }
                } label: {
                    Label("Connect", systemImage: "arrow.clockwise")
                }
                .disabled(viewModel.isBusy)

                Text(viewModel.isBusy ? "Working" : "Ready")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }
}

private struct AssetSection: View {
    @EnvironmentObject private var viewModel: CTViewModel
    @Binding var immersiveOpen: Bool
    let openImmersive: () -> Void
    let closeImmersive: () -> Void

    var body: some View {
        PanelSection(title: "Native 3D Model", systemImage: "cube.transparent") {
            Picker("Asset", selection: Binding(
                get: { viewModel.activeAsset?.id ?? "" },
                set: { newValue in Task { await viewModel.setActiveAsset(newValue) } }
            )) {
                if viewModel.assets.isEmpty {
                    Text("No exported asset").tag("")
                }
                ForEach(viewModel.assets) { asset in
                    Text(asset.id).tag(asset.id)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)

            if let asset = viewModel.activeAsset {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        Text("Quality").foregroundStyle(.secondary)
                        Text(asset.quality)
                    }
                    GridRow {
                        Text("Geometry").foregroundStyle(.secondary)
                        Text("\(asset.meshVertices.formatted()) vertices · \(asset.anatomyCount) organs")
                    }
                    GridRow {
                        Text("Source").foregroundStyle(.secondary)
                        Text(asset.sourceVolumeID)
                    }
                }
                .font(.caption)
            }

            HStack(spacing: 10) {
                Button {
                    Task { await viewModel.exportBalancedAsset() }
                } label: {
                    Label("Export Balanced", systemImage: "square.and.arrow.down")
                }
                .disabled(viewModel.isBusy)

                Button {
                    immersiveOpen ? closeImmersive() : openImmersive()
                } label: {
                    Label(immersiveOpen ? "Close Space" : "Open Space", systemImage: "visionpro")
                }
                .disabled(viewModel.modelFileURL == nil)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }
}

private struct OrganSection: View {
    @EnvironmentObject private var viewModel: CTViewModel
    @State private var searchText = ""
    @State private var expandedGroupIDs: Set<String> = []

    private var filteredAnatomy: [AnatomyItem] {
        guard let activeAsset = viewModel.activeAsset else { return [] }
        let needle = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return activeAsset.anatomy
            .filter { $0.label.lowercased().contains(needle) || $0.group.lowercased().contains(needle) }
            .sorted { $0.label < $1.label }
    }

    private var anatomyGroups: [AnatomyGroup] {
        guard let activeAsset = viewModel.activeAsset else { return [] }
        return Dictionary(grouping: activeAsset.anatomy, by: \.group)
            .map { AnatomyGroup(name: $0.key, items: $0.value.sorted { $0.label < $1.label }) }
            .sorted { lhs, rhs in
                if lhs.items.count == rhs.items.count {
                    return lhs.displayName < rhs.displayName
                }
                return lhs.items.count > rhs.items.count
            }
    }

    var body: some View {
        PanelSection(title: "Organ Controls", systemImage: "checklist") {
            HStack(spacing: 10) {
                MetricPill(value: "\(viewModel.visibleAnatomyLabels.count)", label: "shown")
                MetricPill(value: "\(viewModel.activeAsset?.anatomy.count ?? 0)", label: "detected")
                Spacer()
                Button {
                    if let anatomy = viewModel.activeAsset?.anatomy {
                        if viewModel.visibleAnatomyLabels.count == anatomy.count {
                            viewModel.visibleAnatomyLabels.removeAll()
                        } else {
                            viewModel.visibleAnatomyLabels = Set(anatomy.map(\.label))
                        }
                    }
                } label: {
                    Image(systemName: viewModel.visibleAnatomyLabels.isEmpty ? "eye" : "eye.slash")
                }
                .disabled(viewModel.activeAsset?.anatomy.isEmpty ?? true)
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            ScrollView(.horizontal) {
                HStack(spacing: 8) {
                    ForEach(viewModel.sortedPresetNames, id: \.self) { name in
                        Button(name) {
                            viewModel.applyPreset(name)
                        }
                        .font(.caption.weight(.semibold))
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
                .padding(.vertical, 1)
            }
            .scrollIndicators(.hidden)

            TextField("Search detected anatomy", text: $searchText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .textFieldStyle(.roundedBorder)
                .font(.caption)

            if searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(anatomyGroups) { group in
                            AnatomyGroupCard(
                                group: group,
                                isExpanded: expandedGroupIDs.contains(group.id),
                                toggleExpanded: {
                                    if expandedGroupIDs.contains(group.id) {
                                        expandedGroupIDs.remove(group.id)
                                    } else {
                                        expandedGroupIDs.insert(group.id)
                                    }
                                }
                            )
                        }
                    }
                    .padding(.vertical, 2)
                }
                .frame(minHeight: 470, maxHeight: .infinity)
                .scrollIndicators(.visible)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(filteredAnatomy) { item in
                            OrganToggleRow(item: item)
                        }
                    }
                    .padding(.vertical, 2)
                }
                .frame(minHeight: 470, maxHeight: .infinity)
                .scrollIndicators(.visible)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }
}

private struct AnatomyGroup: Identifiable {
    let name: String
    let items: [AnatomyItem]

    var id: String { name }

    var displayName: String {
        name.replacingOccurrences(of: "_", with: " ").capitalized
    }

    var labels: [String] {
        items.map(\.label)
    }
}

private struct AnatomyGroupCard: View {
    @EnvironmentObject private var viewModel: CTViewModel
    let group: AnatomyGroup
    let isExpanded: Bool
    let toggleExpanded: () -> Void

    private var visibleCount: Int {
        group.labels.filter { viewModel.visibleAnatomyLabels.contains($0) }.count
    }

    private var isFullyVisible: Bool {
        visibleCount == group.items.count && !group.items.isEmpty
    }

    private var isPartiallyVisible: Bool {
        visibleCount > 0 && visibleCount < group.items.count
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Button {
                    toggleExpanded()
                } label: {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.bold))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)

                VStack(alignment: .leading, spacing: 4) {
                    Text(group.displayName)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)
                    Text("\(visibleCount)/\(group.items.count) shown")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 8)

                Button("Only") {
                    viewModel.showOnly(labels: group.labels)
                }
                .font(.caption.weight(.semibold))
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    viewModel.setVisibility(labels: group.labels, visible: !isFullyVisible)
                } label: {
                    Image(systemName: isFullyVisible ? "eye.fill" : isPartiallyVisible ? "eye" : "eye.slash.fill")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(isFullyVisible ? .green : isPartiallyVisible ? .yellow : .secondary)
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(12)
            .background(groupBackground, in: RoundedRectangle(cornerRadius: 8))

            if isExpanded {
                LazyVStack(spacing: 8) {
                    ForEach(group.items) { item in
                        OrganToggleRow(item: item)
                    }
                }
                .padding(.top, 8)
                .padding(.horizontal, 6)
                .padding(.bottom, 8)
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(groupStroke, lineWidth: 1)
        )
    }

    private var groupBackground: Color {
        if isFullyVisible { return Color.green.opacity(0.11) }
        if isPartiallyVisible { return Color.yellow.opacity(0.10) }
        return Color.secondary.opacity(0.08)
    }

    private var groupStroke: Color {
        if isFullyVisible { return Color.green.opacity(0.30) }
        if isPartiallyVisible { return Color.yellow.opacity(0.26) }
        return Color.secondary.opacity(0.12)
    }
}

private struct OrganToggleRow: View {
    @EnvironmentObject private var viewModel: CTViewModel
    let item: AnatomyItem

    private var isVisible: Bool {
        viewModel.visibleAnatomyLabels.contains(item.label)
    }

    var body: some View {
        Button {
            viewModel.toggleAnatomy(item.label)
        } label: {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(item.label.replacingOccurrences(of: "_", with: " ").capitalized)
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(item.group.capitalized)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 8)
                Image(systemName: isVisible ? "eye.fill" : "eye.slash.fill")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(isVisible ? .green : .secondary)
                    .frame(width: 28, height: 28)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isVisible ? Color.green.opacity(0.08) : Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isVisible ? Color.green.opacity(0.24) : Color.secondary.opacity(0.10), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

private struct SliceAnalysisSection: View {
    @EnvironmentObject private var viewModel: CTViewModel

    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12)
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                ModelSnapshotQuestionSection()

                Divider()

                HStack(alignment: .firstTextBaseline) {
                    Label("Slice Analysis", systemImage: "brain")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    Text("\(viewModel.selectedSliceIDs.count)/5 selected")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(viewModel.selectedSliceIDs.isEmpty ? .secondary : .primary)
                }

                TextEditor(text: $viewModel.prompt)
                    .font(.body)
                    .frame(height: 132)
                    .padding(8)
                    .scrollContentBackground(.hidden)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(.secondary.opacity(0.22))
                    )

                HStack(spacing: 10) {
                    Button {
                        Task { await viewModel.analyzeSelectedSlices() }
                    } label: {
                        Label("Run Inference", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(viewModel.selectedSliceIDs.isEmpty || viewModel.isBusy)
                    .buttonStyle(.borderedProminent)

                    Button {
                        viewModel.selectedSliceIDs.removeAll()
                    } label: {
                        Image(systemName: "xmark.circle")
                    }
                    .disabled(viewModel.selectedSliceIDs.isEmpty || viewModel.isBusy)
                    .buttonStyle(.bordered)
                    .help("Clear selected slices")
                }
                .controlSize(.regular)

                if !viewModel.selectedSliceIDs.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(spacing: 8) {
                            ForEach(Array(viewModel.selectedSliceIDs).sorted(), id: \.self) { id in
                                Text(id.replacingOccurrences(of: "ID_", with: ""))
                                    .font(.caption.weight(.medium))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 5)
                                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                    .scrollIndicators(.hidden)
                }

                if !viewModel.analysisText.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Result")
                            .font(.headline)
                        ScrollView {
                            MarkdownTextView(text: viewModel.analysisText)
                        }
                        .frame(maxHeight: 220)
                    }
                    .padding(12)
                    .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
                }

                HStack {
                    Text("Slices")
                        .font(.headline)
                    Spacer()
                    Text("Tap up to five")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(viewModel.slices) { slice in
                        SliceTile(slice: slice)
                    }
                }
                .padding(.bottom, 20)
            }
            .padding(18)
        }
        .background(.regularMaterial)
    }
}

private struct ModelSnapshotQuestionSection: View {
    @EnvironmentObject private var viewModel: CTViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("3D View Question", systemImage: "camera.viewfinder")
                    .font(.title3.weight(.semibold))
                Spacer()
                if viewModel.isModelInteractionLocked {
                    Label("Frozen", systemImage: "lock.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.green)
                }
            }

            Text(viewModel.snapshotStatusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)

            TextEditor(text: $viewModel.snapshotQuestion)
                .font(.body)
                .frame(height: 82)
                .padding(8)
                .scrollContentBackground(.hidden)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.secondary.opacity(0.20))
                )

            HStack(spacing: 10) {
                Button {
                    viewModel.captureCurrent3DView()
                } label: {
                    Label(viewModel.capturedSnapshotBase64 == nil ? "Capture" : "Recapture", systemImage: "camera")
                        .frame(maxWidth: .infinity)
                }
                .disabled(viewModel.isBusy)
                .buttonStyle(.bordered)

                Button {
                    Task { await viewModel.analyzeCurrent3DView() }
                } label: {
                    Label("Ask Gemini", systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .disabled(viewModel.isBusy || viewModel.snapshotQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .buttonStyle(.borderedProminent)

                Button {
                    viewModel.releaseCurrent3DView()
                } label: {
                    Image(systemName: "lock.open")
                }
                .disabled(viewModel.capturedSnapshotBase64 == nil || viewModel.isBusy)
                .buttonStyle(.bordered)
                .help("Release frozen view")
            }
            .controlSize(.regular)

            SnapshotAnswerPanel()
        }
        .padding(12)
        .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct SnapshotAnswerPanel: View {
    @EnvironmentObject private var viewModel: CTViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Label("3D Answer", systemImage: "text.bubble")
                    .font(.headline)
                Spacer()
                if !viewModel.snapshotAnalysisText.isEmpty {
                    Button {
                        viewModel.snapshotAnalysisText = ""
                    } label: {
                        Image(systemName: "xmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Clear 3D answer")
                }
            }

            if viewModel.snapshotAnalysisText.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("No 3D answer yet")
                        .font(.callout.weight(.semibold))
                    Text("Capture a view, ask Gemini, and the response will appear here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
                .padding(12)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            } else {
                ScrollView {
                    MarkdownTextView(text: viewModel.snapshotAnalysisText)
                        .padding(.trailing, 8)
                }
                .frame(minHeight: 190, maxHeight: 310)
                .padding(12)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.accentColor.opacity(0.30), lineWidth: 1)
                )
            }
        }
    }
}

private struct MarkdownTextView: View {
    let text: String

    var body: some View {
        let formattedText = Self.formattedClinicalText(text)

        Group {
            if let attributed = try? AttributedString(markdown: formattedText) {
                Text(attributed)
            } else {
                Text(formattedText)
            }
        }
        .font(.body)
        .lineSpacing(4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }

    private static func formattedClinicalText(_ rawText: String) -> String {
        var output = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !output.isEmpty else { return output }

        let sectionHeadings = [
            "Direct Answer",
            "Visible Anatomy",
            "Spatial Observations",
            "Uncertainty and Limitations",
            "Suggested Next View or Slice to Inspect",
            "Anatomy Visible",
            "Key Observations",
            "Potential Abnormalities Or Issues",
            "Limitations",
            "Suggested Follow-Up Questions For A Clinician"
        ]

        for heading in sectionHeadings {
            let escaped = NSRegularExpression.escapedPattern(for: heading)
            let pattern = "(^|\\n+|(?<=[.!?])\\s*)(?:#{1,6}\\s*)?\(escaped):?\\s*"
            output = replaceRegex(
                pattern,
                in: output,
                with: "\n\n## \(NSRegularExpression.escapedTemplate(for: heading))\n\n"
            )
        }

        let detailLabels = [
            "Airway",
            "Arteries",
            "Arterial System",
            "Veins",
            "Venous System",
            "Lungs",
            "Cardiac Structures",
            "Bones",
            "Vascular Architecture",
            "Venous Convergence",
            "Thoracic Volume",
            "Mediastinal Centering",
            "Anatomical Integration",
            "Scale",
            "Visual Assessment",
            "Pixel-Level Confirmation",
            "Diagnostic Constraints",
            "Anatomical Scope",
            "Non-Diagnostic",
            "Axial Source Slices",
            "Coronal Reformation",
            "Coronal Multiplanar Reconstruction",
            "Radiologist Review"
        ]

        for label in detailLabels {
            let escaped = NSRegularExpression.escapedPattern(for: label)
            let pattern = "(^|\\n+|(?<=[.!?])\\s*)\(escaped):\\s*"
            output = replaceRegex(
                pattern,
                in: output,
                with: "\n\n- **\(NSRegularExpression.escapedTemplate(for: label)):** "
            )
        }

        output = replaceRegex("\\n{3,}", in: output, with: "\n\n")
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func replaceRegex(_ pattern: String, in text: String, with replacement: String) -> String {
        guard let expression = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return text
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return expression.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement)
    }
}

private struct SliceTile: View {
    @EnvironmentObject private var viewModel: CTViewModel
    let slice: SliceRecord

    var selected: Bool {
        viewModel.selectedSliceIDs.contains(slice.id)
    }

    var body: some View {
        Button {
            viewModel.toggleSlice(slice.id)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                ZStack(alignment: .topTrailing) {
                    AsyncImage(url: viewModel.resolvedURL(slice.thumbnailURL)) { phase in
                        switch phase {
                        case let .success(image):
                            image
                                .resizable()
                                .scaledToFill()
                        case .failure:
                            Image(systemName: "photo")
                                .font(.largeTitle)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        default:
                            ProgressView()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                    }
                    .frame(height: 128)
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                    if selected {
                        Image(systemName: "checkmark.circle.fill")
                            .symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .green)
                            .padding(6)
                    }
                }

                Text(slice.displayName)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                Text(slice.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .frame(maxWidth: .infinity, minHeight: 186, alignment: .topLeading)
            .background(selected ? Color.accentColor.opacity(0.26) : Color.secondary.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(selected ? Color.accentColor.opacity(0.9) : Color.secondary.opacity(0.16), lineWidth: selected ? 2 : 1)
            )
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: 8))
    }
}

#Preview("CT Vision App", windowStyle: .plain) {
    ContentView()
        .environmentObject(CTViewModel.preview)
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(CTViewModel.preview)
            .previewDisplayName("CT Vision App")
    }
}

@MainActor
private extension CTViewModel {
    static var preview: CTViewModel {
        let viewModel = CTViewModel()
        viewModel.statusText = "Preview data loaded."
        viewModel.slices = [
            SliceRecord(
                id: "ID_104_CT",
                age: 54,
                contrast: true,
                dicomName: "arterial_chest_104.dcm",
                thumbnailURL: "/preview/chest-104.png"
            ),
            SliceRecord(
                id: "ID_118_CT",
                age: 61,
                contrast: false,
                dicomName: "noncontrast_chest_118.dcm",
                thumbnailURL: "/preview/chest-118.png"
            )
        ]
        viewModel.assets = [Self.previewAsset]
        viewModel.activeAssetID = Self.previewAsset.id
        viewModel.visibleAnatomyLabels = Set(Self.previewAsset.anatomy.map(\.label))
        return viewModel
    }

    static var previewAsset: VisionAssetManifest {
        let anatomy = [
            AnatomyItem(label: "heart", group: "cardiac", color: [210, 58, 72], vertices: 18_420, faces: 36_804),
            AnatomyItem(label: "left_lung", group: "respiratory", color: [84, 164, 214], vertices: 24_118, faces: 48_036),
            AnatomyItem(label: "right_lung", group: "respiratory", color: [91, 178, 205], vertices: 25_002, faces: 49_880),
            AnatomyItem(label: "aorta", group: "vessels", color: [229, 77, 67], vertices: 8_612, faces: 17_144),
            AnatomyItem(label: "rib_cage", group: "bones", color: [232, 221, 198], vertices: 31_450, faces: 62_704)
        ]

        return VisionAssetManifest(
            id: "totalseg_CT_chest_preview_balanced",
            kind: "usdz",
            quality: "balanced",
            sourceVolumeID: "totalseg_CT_chest_preview",
            modelURL: "/preview/model.usdz",
            usdzURL: "/preview/model.usdz",
            glbURL: nil,
            meshVertices: anatomy.reduce(0) { $0 + $1.vertices },
            meshFaces: anatomy.reduce(0) { $0 + $1.faces },
            anatomyCount: anatomy.count,
            anatomy: anatomy,
            presets: [
                "All": anatomy.map(\.label),
                "Chest Core": ["heart", "left_lung", "right_lung", "aorta"],
                "Heart": ["heart", "aorta"],
                "Lungs": ["left_lung", "right_lung"],
                "Bones": ["rib_cage"]
            ]
        )
    }
}
