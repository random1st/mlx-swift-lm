//
//  GatedDeltaSpeculativeRound.swift
//  mlx-swift-lm
//
//  Capture and rollback for a speculative verification round on a hybrid
//  (gated-DeltaNet + full attention) backbone.
//

import Foundation
import MLX

/// One gated-DeltaNet layer's inputs to a single verification round.
///
/// A verify pass runs `[anchor] + drafts` through the target in one forward. When
/// only `keep` of those positions survive, the attention layers can be trimmed —
/// their K/V rows are position-local — but the recurrent state cannot: it is an
/// accumulation over all `width` positions. Re-running the whole model over the
/// accepted prefix would cost a second sweep over the weights and erase the
/// speedup, so the round's recurrence inputs are stashed instead: the gated-delta
/// update is causal, so the state after `keep` positions is a function of the
/// pre-round state and inputs `[0, keep)` alone, and replaying just the recurrence
/// over those few positions reproduces a committed forward.
public struct GatedDeltaRoundCapture {
    /// Post-norm queries `[B, S, Hk, Dk]`.
    public let q: MLXArray
    /// Post-norm keys `[B, S, Hk, Dk]`.
    public let k: MLXArray
    /// Values `[B, S, Hv, Dv]`.
    public let v: MLXArray
    /// Decay-gate input `[B, S, Hv]`.
    public let a: MLXArray
    /// Beta-gate input `[B, S, Hv]`.
    public let b: MLXArray
    /// Layer parameter, position-independent.
    public let aLog: MLXArray
    /// Layer parameter, position-independent.
    public let dtBias: MLXArray
    /// Recurrent state as it stood *before* the round; `nil` when the layer had
    /// no state yet (the replay then starts from zeros, as the round itself did).
    public let state: MLXArray?
    /// Optional per-position validity mask `[B, S]`.
    public let mask: MLXArray?
    /// `[pre-round conv window ‖ round conv input]`, `[B, (kernel - 1) + S, D]`.
    /// The committed window after `keep` positions is a slice of this, because
    /// each row depends only on its own position.
    public let convInput: MLXArray
    /// `kernel - 1`: how many rows the conv window keeps.
    public let convWindow: Int

    public init(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        a: MLXArray,
        b: MLXArray,
        aLog: MLXArray,
        dtBias: MLXArray,
        state: MLXArray?,
        mask: MLXArray?,
        convInput: MLXArray,
        convWindow: Int
    ) {
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
        self.state = state
        self.mask = mask
        self.convInput = convInput
        self.convWindow = convWindow
    }

    /// The recurrent state a committed forward over the first `keep` positions of
    /// this round would have left behind.
    public func recurrentState(keep: Int) -> MLXArray {
        let (_, rebuilt) = gatedDeltaUpdate(
            q: q[0..., ..<keep, 0..., 0...],
            k: k[0..., ..<keep, 0..., 0...],
            v: v[0..., ..<keep, 0..., 0...],
            a: a[0..., ..<keep, 0...],
            b: b[0..., ..<keep, 0...],
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask.map { $0[0..., ..<keep] })
        return rebuilt
    }

    /// The conv window a committed forward over the first `keep` positions would
    /// have kept: the `kernel - 1` rows ending at position `keep`.
    public func convState(keep: Int) -> MLXArray {
        guard convWindow > 0 else {
            return MLXArray.zeros([convInput.dim(0), 0, convInput.dim(2)], dtype: convInput.dtype)
        }
        return contiguous(convInput[0..., keep ..< (keep + convWindow), 0...])
    }
}

/// Collects one round's ``GatedDeltaRoundCapture`` in layer order.
///
/// A reference type so a single forward pass can fill it through the layer chain
/// without either a mutable property on the model — the drafter path deliberately
/// keeps no transient state there — or an `inout` parameter on every call site.
public final class GatedDeltaRoundCaptureSink {
    public private(set) var captures: [GatedDeltaRoundCapture] = []

    public init() {}

    public func append(_ capture: GatedDeltaRoundCapture) {
        captures.append(capture)
    }
}

/// Restore a hybrid model's caches to the state a committed forward over the
/// first `keep` positions of the last verification round would have left.
///
/// - Parameters:
///   - cache: the model's cache array, in layer order.
///   - captures: one capture per recurrent layer, in layer order, from the verify
///     pass that wrote `width` positions.
///   - width: how many positions that verify pass wrote (`1 + drafts`).
///   - keep: how many of them are kept (`1 + accepted`), at least 1.
///
/// Recurrent entries are rebuilt from the stashed pre-round state; every other
/// trimmable entry drops its rejected tail, which is exact because K/V rows are
/// position-local.
public func rollbackGatedDeltaRound(
    cache: [any KVCache],
    captures: [GatedDeltaRoundCapture],
    width: Int,
    keep: Int
) {
    precondition(keep >= 1 && keep <= width, "keep \(keep) outside 1...\(width)")
    guard keep < width else { return }

    var captureIndex = 0
    for entry in cache {
        guard let recurrent = entry as? MambaCache else {
            if entry.isTrimmable {
                entry.trim(width - keep)
            }
            continue
        }
        // A recurrent layer without a capture cannot be rebuilt, and leaving it
        // holding the rejected tail would silently corrupt every later token.
        precondition(
            captureIndex < captures.count,
            "rollback got \(captures.count) gated-delta captures for more recurrent caches")
        let capture = captures[captureIndex]
        captureIndex += 1

        recurrent[0] = capture.convState(keep: keep)
        recurrent[1] = capture.recurrentState(keep: keep)
        // The round advanced the cache by its full width; only `keep` positions
        // were actually consumed.
        recurrent.advance(-(width - keep))
    }
}
