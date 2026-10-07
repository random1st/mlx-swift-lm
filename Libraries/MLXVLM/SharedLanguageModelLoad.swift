import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum SharedLanguageModelLoadError: LocalizedError {
    case notShared([String])

    public var errorDescription: String? {
        switch self {
        case .notShared(let keys):
            return "\(keys.count) language-model weights have no resident counterpart "
                + "(first: \(keys.prefix(3).joined(separator: ", "))); loading them would "
                + "put a second copy of the model in memory"
        }
    }
}

extension VLMModelFactory {
    /// Loads the vision model of a checkpoint whose language model is already resident as a
    /// text model, reusing that model's arrays instead of reading them again.
    ///
    /// A multimodal checkpoint is a vision tower plus a language model, and a text-only
    /// loader reads the second and skips the first. Loading the whole thing again for
    /// images would hold the language model twice - 20 GB for a 35B MoE. Here only the
    /// tower is read from disk; every `language_model.*` parameter is the resident text
    /// model's own array, so the two share storage and the cost of vision is the tower.
    ///
    /// The vision model is left unprepared: preparing would fuse projections into new
    /// arrays, which is a copy. It runs the unfused path, which is correct and slightly
    /// slower, on the requests that carry an image.
    ///
    /// - Parameters:
    ///   - directory: the checkpoint the text model was loaded from.
    ///   - languageModel: the resident model; its parameter paths must match the vision
    ///     model's `language_model` subtree, as they do for the text and vision Qwen 3.5.
    ///   - tokenizer: the resident model's tokenizer.
    /// - Throws: ``SharedLanguageModelLoadError/notShared(_:)`` if any language-model
    ///   weight would have to come from disk.
    public func loadSharingLanguageModel(
        directory: URL, languageModel: Module, tokenizer: any Tokenizer,
        configuration textConfiguration: ModelConfiguration
    ) async throws -> ModelContext {
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let model = try await typeRegistry.createModel(
            configuration: configData, modelType: baseConfig.modelType)

        // Only the tower is read from disk, by byte range. The library's loader reads whole
        // shards, which for a 35B MoE put a second 20 GB copy of the language model in
        // memory for the length of the load before it could be dropped.
        let (tower, metadata) = try readTensors(in: directory) { !$0.hasPrefix("language_model.") }
        var weights = tower
        for (key, array) in languageModel.parameters().flattened()
        where key.hasPrefix("language_model.") {
            weights[key] = array
        }
        weights = model.sanitize(weights: weights, metadata: metadata)

        if let quantization = baseConfig.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil
                    ? quantization.quantization(layer: path)?.asTuple : nil
            }
        }

        // `verify: .all` is what proves the sharing: a vision-model parameter the resident
        // text model does not have fails here instead of being read from disk.
        do {
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        } catch {
            throw SharedLanguageModelLoadError.notShared([String(describing: error)])
        }
        eval(Array(tower.values))

        let processorConfiguration = try await resolveProcessorConfiguration(
            from: directory,
            context: VLMProcessorLoadingContext(
                modelId: textConfiguration.name, modelType: baseConfig.modelType,
                configurationData: configData),
            registry: processorLoadingRegistry)
        let processor = try await processorRegistry.createModel(
            configuration: processorConfiguration.data,
            processorType: processorConfiguration.processorType, tokenizer: tokenizer)
        return ModelContext(
            configuration: textConfiguration, model: model, processor: processor,
            tokenizer: tokenizer)
    }
}

/// The tensors of a safetensors checkpoint whose names pass `include`, each read from its
/// own byte range; the rest of every file is never touched.
func readTensors(
    in directory: URL, where include: (String) -> Bool
) throws -> ([String: MLXArray], [String: String]) {
    let files = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "safetensors" }
    var arrays: [String: MLXArray] = [:]
    var metadata: [String: String] = [:]
    for file in files {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else { continue }
        let headerLength = lengthBytes.withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) }
        guard let headerData = try handle.read(upToCount: headerLength),
            let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { continue }
        let base = UInt64(8 + headerLength)
        for (name, value) in header {
            if name == "__metadata__" {
                for (k, v) in value as? [String: String] ?? [:] { metadata[k] = v }
                continue
            }
            guard include(name), let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String, let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { continue }
            try handle.seek(toOffset: base + UInt64(offsets[0]))
            let bytes = try handle.read(upToCount: offsets[1] - offsets[0]) ?? Data()
            arrays[name] = try makeArray(bytes, shape: shape, dtype: dtype, name: name)
        }
    }
    return (arrays, metadata)
}

private func makeArray(_ data: Data, shape: [Int], dtype: String, name: String) throws -> MLXArray {
    switch dtype {
    case "BF16": return MLXArray(data, shape, type: UInt16.self).view(dtype: .bfloat16)
    case "F16": return MLXArray(data, shape, type: UInt16.self).view(dtype: .float16)
    case "F32": return MLXArray(data, shape, type: Float.self)
    case "U32": return MLXArray(data, shape, type: UInt32.self)
    case "I32": return MLXArray(data, shape, type: Int32.self)
    case "U8": return MLXArray(data, shape, type: UInt8.self)
    default:
        throw NSError(domain: "SharedLanguageModelLoad", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(name): safetensors dtype \(dtype) is not supported here"
        ])
    }
}
