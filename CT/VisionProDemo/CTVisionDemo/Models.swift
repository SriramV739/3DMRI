import Foundation

struct SliceRecord: Identifiable, Decodable, Hashable {
    let id: String
    let age: Int?
    let contrast: Bool
    let dicomName: String?
    let thumbnailURL: String

    init(id: String, age: Int?, contrast: Bool, dicomName: String?, thumbnailURL: String) {
        self.id = id
        self.age = age
        self.contrast = contrast
        self.dicomName = dicomName
        self.thumbnailURL = thumbnailURL
    }

    enum CodingKeys: String, CodingKey {
        case id
        case age
        case contrast
        case dicomName = "dicom_name"
        case thumbnailURL = "thumbnail_url"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        dicomName = try container.decodeIfPresent(String.self, forKey: .dicomName)
        thumbnailURL = try container.decode(String.self, forKey: .thumbnailURL)
        age = Self.decodeFlexibleInt(container, forKey: .age)
        contrast = Self.decodeFlexibleBool(container, forKey: .contrast)
    }

    private static func decodeFlexibleInt(_ container: KeyedDecodingContainer<CodingKeys>, forKey key: CodingKeys) -> Int? {
        if let value = try? container.decodeIfPresent(Int.self, forKey: key) {
            return value
        }
        if let value = try? container.decodeIfPresent(String.self, forKey: key) {
            return Int(value)
        }
        return nil
    }

    private static func decodeFlexibleBool(_ container: KeyedDecodingContainer<CodingKeys>, forKey key: CodingKeys) -> Bool {
        if let value = try? container.decodeIfPresent(Bool.self, forKey: key) {
            return value
        }
        if let value = try? container.decodeIfPresent(Int.self, forKey: key) {
            return value != 0
        }
        if let value = try? container.decodeIfPresent(String.self, forKey: key) {
            return ["1", "true", "yes", "contrast"].contains(value.lowercased())
        }
        return false
    }

    var displayName: String {
        id.replacingOccurrences(of: "ID_", with: "").replacingOccurrences(of: "_CT", with: "")
    }

    var subtitle: String {
        let contrastText = contrast ? "contrast" : "noncontrast"
        if let age {
            return "\(contrastText) · age \(age)"
        }
        return contrastText
    }
}

struct AnatomyItem: Identifiable, Decodable, Hashable {
    var id: String { label }
    let label: String
    let group: String
    let color: [Int]
    let vertices: Int
    let faces: Int
}

struct VisionAssetManifest: Identifiable, Decodable, Hashable {
    let id: String
    let kind: String
    let quality: String
    let sourceVolumeID: String
    let modelURL: String
    let usdzURL: String
    let glbURL: String?
    let meshVertices: Int
    let meshFaces: Int
    let anatomyCount: Int
    let anatomy: [AnatomyItem]
    let presets: [String: [String]]

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case quality
        case sourceVolumeID = "source_volume_id"
        case modelURL = "model_url"
        case usdzURL = "usdz_url"
        case glbURL = "glb_url"
        case meshVertices = "mesh_vertices"
        case meshFaces = "mesh_faces"
        case anatomyCount = "anatomy_count"
        case anatomy
        case presets
    }
}

struct AnalyzeRequest: Encodable {
    let sliceIDs: [String]
    let userNote: String
    let dryRun: Bool
    let provider: String

    enum CodingKeys: String, CodingKey {
        case sliceIDs = "slice_ids"
        case userNote = "user_note"
        case dryRun = "dry_run"
        case provider
    }
}

struct AnalyzeResponse: Decodable {
    let mode: String
    let provider: String?
    let model: String
    let analysis: String
}

struct SnapshotAnalyzeRequest: Encodable {
    let imageBase64: String
    let userNote: String
    let assetID: String?
    let visibleLabels: [String]
    let rotationX: Float
    let rotationY: Float
    let scale: Float
    let dryRun: Bool

    enum CodingKeys: String, CodingKey {
        case imageBase64 = "image_base64"
        case userNote = "user_note"
        case assetID = "asset_id"
        case visibleLabels = "visible_labels"
        case rotationX = "rotation_x"
        case rotationY = "rotation_y"
        case scale
        case dryRun = "dry_run"
    }
}
