import RealityKit
import SwiftUI

struct RealityModelView: View {
    @EnvironmentObject private var viewModel: CTViewModel
    let immersive: Bool
    let resetToken: Int

    @State private var rootEntity = Entity()
    @State private var loadedModelRoot: Entity?
    @State private var rotationX: Float = -0.12
    @State private var rotationY: Float = 0.0
    @State private var settledRotationX: Float = -0.12
    @State private var settledRotationY: Float = 0.0
    @State private var modelScale: Float = 1.0
    @State private var settledModelScale: Float = 1.0
    @State private var assetNormalizationScale: Float = 1.0

    private var defaultModelScale: Float {
        1.0
    }

    private var targetModelSize: Float {
        immersive ? 0.95 : 0.30
    }

    private var modelPosition: SIMD3<Float> {
        immersive ? [0, -0.05, -0.75] : [0.04, 0.07, 0]
    }

    var body: some View {
        ZStack {
            RealityView { content in
                content.add(rootEntity)
            } update: { _ in
                rootEntity.orientation = simd_quatf(angle: rotationY, axis: [0, 1, 0]) * simd_quatf(angle: rotationX, axis: [1, 0, 0])
                rootEntity.scale = SIMD3<Float>(repeating: assetNormalizationScale * modelScale)
                rootEntity.position = modelPosition
                applyVisibility()
            }
            .clipped()
            .task(id: viewModel.modelFileURL) {
                await loadCurrentModel()
            }
            .gesture(
                DragGesture()
                    .onChanged { value in
                        guard !viewModel.isModelInteractionLocked else { return }
                        rotationY = settledRotationY + Float(value.translation.width) * 0.004
                        rotationX = settledRotationX + Float(value.translation.height) * 0.003
                        viewModel.updateModelViewState(rotationX: rotationX, rotationY: rotationY, scale: modelScale)
                    }
                    .onEnded { _ in
                        guard !viewModel.isModelInteractionLocked else { return }
                        settledRotationX = rotationX
                        settledRotationY = rotationY
                        viewModel.updateModelViewState(rotationX: rotationX, rotationY: rotationY, scale: modelScale)
                    }
            )
            .simultaneousGesture(
                MagnificationGesture()
                    .onChanged { value in
                        guard !viewModel.isModelInteractionLocked else { return }
                        modelScale = clampedScale(settledModelScale * Float(value))
                        viewModel.updateModelViewState(rotationX: rotationX, rotationY: rotationY, scale: modelScale)
                    }
                    .onEnded { _ in
                        guard !viewModel.isModelInteractionLocked else { return }
                        settledModelScale = modelScale
                        viewModel.updateModelViewState(rotationX: rotationX, rotationY: rotationY, scale: modelScale)
                    }
            )
            .simultaneousGesture(
                TapGesture()
                    .targetedToAnyEntity()
                    .onEnded { value in
                        guard !viewModel.isModelInteractionLocked else { return }
                        viewModel.selectedOrganName = readableName(for: value.entity)
                    }
            )
            if viewModel.isModelInteractionLocked {
                VStack {
                    Spacer()
                    Label("Frozen snapshot view", systemImage: "lock.fill")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
                        .padding(.bottom, 18)
                }
            }
        }
        .onAppear {
            resetViewTransform()
        }
        .onChange(of: resetToken) {
            resetViewTransform()
        }
        .frame(minWidth: immersive ? 900 : 760, minHeight: immersive ? 620 : 760)
    }

    private func clampedScale(_ value: Float) -> Float {
        min(2.6, max(0.75, value))
    }

    private func resetViewTransform() {
        rotationX = -0.12
        rotationY = 0
        settledRotationX = rotationX
        settledRotationY = rotationY
        modelScale = defaultModelScale
        settledModelScale = modelScale
        viewModel.updateModelViewState(rotationX: rotationX, rotationY: rotationY, scale: modelScale)
    }

    private func loadCurrentModel() async {
        guard let fileURL = viewModel.modelFileURL else {
            rootEntity.children.removeAll()
            loadedModelRoot = nil
            return
        }

        do {
            let entity = try await Entity(contentsOf: fileURL)
            entity.name = "CTAnatomy"
            prepareForSelection(entity)
            makeModelMoreOpaque(entity)
            let normalizedScale = normalizedScale(for: entity)
            centerModel(entity)
            await MainActor.run {
                rootEntity.children.removeAll()
                assetNormalizationScale = normalizedScale
                rootEntity.addChild(entity)
                loadedModelRoot = entity
                applyVisibility()
            }
        } catch {
            await MainActor.run {
                viewModel.statusText = "Could not load native model: \(error.localizedDescription)"
            }
        }
    }

    private func normalizedScale(for entity: Entity) -> Float {
        let bounds = entity.visualBounds(relativeTo: nil)
        let largestExtent = max(bounds.extents.x, bounds.extents.y, bounds.extents.z)
        guard largestExtent > 0 else { return 1.0 }
        return targetModelSize / largestExtent
    }

    private func centerModel(_ entity: Entity) {
        let bounds = entity.visualBounds(relativeTo: nil)
        entity.position -= bounds.center
    }

    private func prepareForSelection(_ entity: Entity) {
        addInputTargets(entity)
        entity.generateCollisionShapes(recursive: true)
    }

    private func makeModelMoreOpaque(_ entity: Entity) {
        entity.components.set(OpacityComponent(opacity: 0.96))

        if let modelEntity = entity as? ModelEntity, var model = modelEntity.model {
            model.materials = model.materials.map { material in
                guard var physicallyBased = material as? PhysicallyBasedMaterial else {
                    return material
                }

                physicallyBased.blending = .transparent(
                    opacity: PhysicallyBasedMaterial.Opacity(floatLiteral: 0.9)
                )
                return physicallyBased
            }
            modelEntity.model = model
        }

        for child in entity.children {
            makeModelMoreOpaque(child)
        }
    }

    private func addInputTargets(_ entity: Entity) {
        entity.components.set(InputTargetComponent())
        for child in entity.children {
            addInputTargets(child)
        }
    }

    private func applyVisibility() {
        guard let loadedModelRoot else { return }
        let allLabels = Set(viewModel.activeAsset?.anatomy.map(\.label) ?? [])
        applyVisibilityRecursively(loadedModelRoot, allLabels: allLabels)
    }

    private func applyVisibilityRecursively(_ entity: Entity, allLabels: Set<String>) {
        if allLabels.contains(entity.name) {
            entity.isEnabled = viewModel.visibleAnatomyLabels.contains(entity.name)
        }
        for child in entity.children {
            applyVisibilityRecursively(child, allLabels: allLabels)
        }
    }

    private func readableName(for entity: Entity) -> String {
        var current: Entity? = entity
        let allLabels = Set(viewModel.activeAsset?.anatomy.map(\.label) ?? [])
        while let candidate = current {
            if allLabels.contains(candidate.name) {
                return candidate.name.replacingOccurrences(of: "_", with: " ").capitalized
            }
            current = candidate.parent
        }
        return entity.name.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

struct ImmersiveVolumeView: View {
    var body: some View {
        RealityModelView(immersive: true, resetToken: 0)
    }
}
