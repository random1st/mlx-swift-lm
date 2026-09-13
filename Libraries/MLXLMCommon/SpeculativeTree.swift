//
//  SpeculativeTree.swift
//  mlx-swift-lm
//
//  A speculative round whose drafts form a tree rather than a chain.
//

import Foundation
import MLX

/// The shape of one tree-verified round.
///
/// Rows are the verify pass's positions: row 0 is the anchor and every other row
/// names the row it continues. A parent always precedes its children, which is
/// what lets attention over the tree be a mask over rows already written and the
/// recurrent layers be a batch of root paths.
public struct SpeculativeTreePlan {
    /// Parent row per row; `-1` for the anchor.
    public let parents: [Int]
    /// Distance from the anchor per row; the anchor is 0.
    public let depths: [Int]
    /// Root paths, one per leaf, each starting at the anchor row.
    public let paths: [[Int]]
    /// The longest path, in rows: what every path is padded to for the batched
    /// recurrence.
    public let pathLength: Int
    /// Where each row's recurrent output lives once the paths have been run as a
    /// batch: the first path that contains the row, and the row's depth in it.
    public let rowSources: [(path: Int, depth: Int)]

    public var rowCount: Int { parents.count }
    public var pathCount: Int { paths.count }

    public init(parents: [Int]) {
        precondition(parents.first == -1, "row 0 is the anchor and has no parent")
        for (row, parent) in parents.enumerated().dropFirst() {
            precondition(
                parent >= 0 && parent < row,
                "row \(row) continues row \(parent), which must precede it")
        }
        self.parents = parents

        var depths = [Int](repeating: 0, count: parents.count)
        var hasChild = [Bool](repeating: false, count: parents.count)
        for row in 1 ..< parents.count {
            depths[row] = depths[parents[row]] + 1
            hasChild[parents[row]] = true
        }
        self.depths = depths

        var paths: [[Int]] = []
        var sources = [(path: Int, depth: Int)?](repeating: nil, count: parents.count)
        for leaf in 0 ..< parents.count where !hasChild[leaf] {
            var path: [Int] = []
            var row = leaf
            while row >= 0 {
                path.append(row)
                row = parents[row]
            }
            path.reverse()
            for (depth, row) in path.enumerated() where sources[row] == nil {
                sources[row] = (paths.count, depth)
            }
            paths.append(path)
        }
        self.paths = paths
        self.pathLength = paths.map(\.count).max() ?? 1
        self.rowSources = sources.map { $0! }
    }

    /// A chain of `rows` positions: the degenerate tree every round is today.
    public static func chain(rows: Int) -> SpeculativeTreePlan {
        SpeculativeTreePlan(parents: [-1] + (1 ..< max(rows, 1)).map { $0 - 1 })
    }

    /// Whether `rows` is a root path of this tree, in order, starting at the anchor.
    public func isRootPath(_ rows: [Int]) -> Bool {
        guard rows.first == 0 else { return false }
        for (index, row) in rows.enumerated().dropFirst() where parents[row] != rows[index - 1] {
            return false
        }
        return true
    }

    /// `[rows, prefix + rows]` boolean attention mask: every row sees the whole
    /// cached prefix and, among this round's rows, only its own root path.
    public func attentionMask(prefix: Int) -> MLXArray {
        let rows = rowCount
        var ancestry = [Int32](repeating: 0, count: rows * rows)
        for row in 0 ..< rows {
            if row > 0 {
                let parent = parents[row]
                for column in 0 ..< rows {
                    ancestry[row * rows + column] = ancestry[parent * rows + column]
                }
            }
            ancestry[row * rows + row] = 1
        }
        let tree = MLXArray(ancestry).reshaped(rows, rows).asType(.bool)
        guard prefix > 0 else { return tree }
        return concatenated([MLXArray.ones([rows, prefix], dtype: .bool), tree], axis: 1)
    }

    /// Position of every row: the cache offset plus the row's depth.
    public func positions(offset: Int) -> MLXArray {
        MLXArray(depths.map { Int32(offset + $0) })
    }

    /// `[paths, pathLength]` row indices, each path padded with the anchor row.
    ///
    /// Padding repeats row 0, so a padded slot recomputes the anchor's own update
    /// on top of a finished path. Nothing reads those slots - ``rowSourceIndices()``
    /// never points at one - and the recurrent state the round leaves behind is
    /// replaced by the rollback, so the garbage stays inside the round.
    public func pathRowIndices() -> MLXArray {
        var flat: [Int32] = []
        flat.reserveCapacity(paths.count * pathLength)
        for path in paths {
            flat.append(contentsOf: path.map(Int32.init))
            flat.append(contentsOf: repeatElement(Int32(0), count: pathLength - path.count))
        }
        return MLXArray(flat).reshaped(paths.count, pathLength)
    }

    /// `[rows]` indices into a `[paths * pathLength, ...]` batch: each row's own
    /// (path, depth) slot, for scattering the path-major recurrent output back
    /// into row order.
    public func rowSourceIndices() -> MLXArray {
        MLXArray(rowSources.map { Int32($0.path * pathLength + $0.depth) })
    }
}

/// Describes the verify pass's rows as a tree. The target then masks attention to
/// root paths, positions rows by depth, and runs the recurrent layers over every
/// root path. A pass that sets this must be followed by
/// ``rollbackGatedDeltaTree(cache:captures:width:keepRows:)``: until then the
/// recurrent caches hold an arbitrary path's state.
public let mtpTreePlanKey = LMOutput.Key<SpeculativeTreePlan>("mtp.treePlan")

extension GatedDeltaRoundCapture {
    /// The recurrent state a committed forward over `rows` - a root path of the
    /// tree, anchor first - would have left behind.
    ///
    /// The chain case slices a prefix; here the same recurrence runs over gathered
    /// rows, which is the same thing because the gated-delta update reads each
    /// position only through the state its predecessors built.
    public func recurrentState(rows: [Int]) -> MLXArray {
        let index = MLXArray(rows.map(Int32.init))
        let (_, rebuilt) = gatedDeltaUpdate(
            q: take(q, index, axis: 1),
            k: take(k, index, axis: 1),
            v: take(v, index, axis: 1),
            a: take(a, index, axis: 1),
            b: take(b, index, axis: 1),
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask.map { take($0, index, axis: 1) })
        return rebuilt
    }

    /// The conv window a committed forward over `rows` would have kept: the last
    /// `convWindow` rows of `[pre-round window ‖ the path's conv input]`.
    public func convState(rows: [Int]) -> MLXArray {
        guard convWindow > 0 else {
            return MLXArray.zeros([convInput.dim(0), 0, convInput.dim(2)], dtype: convInput.dtype)
        }
        let picked = take(convInput, MLXArray(rows.map { Int32(convWindow + $0) }), axis: 1)
        let committed = concatenated([convInput[0..., ..<convWindow, 0...], picked], axis: 1)
        return contiguous(committed[0..., (committed.dim(1) - convWindow)..., 0...])
    }
}

/// Restore a hybrid model's caches to the state a committed forward over
/// `keepRows` - a root path of the last tree-verified round, anchor first - would
/// have left.
///
/// Recurrent entries are rebuilt from the stashed pre-round inputs along the path.
/// Attention entries keep the path's K/V rows and drop the rest: rows are
/// position-local, so moving the kept ones down to the front of the round is
/// exact. A path that happens to be a prefix of the rows takes the cheaper
/// trim-the-tail route, which is also what every chain round does.
public func rollbackGatedDeltaTree(
    cache: [any KVCache],
    captures: [GatedDeltaRoundCapture],
    width: Int,
    keepRows: [Int]
) {
    let keep = keepRows.count
    precondition(keep >= 1 && keep <= width, "keep \(keep) rows outside 1...\(width)")
    precondition(keepRows.first == 0, "a kept path starts at the anchor row")
    let isPrefix = keepRows == Array(0 ..< keep)
    if isPrefix && keep == width { return }

    var captureIndex = 0
    for entry in cache {
        guard let recurrent = entry as? MambaCache else {
            if isPrefix {
                if entry.isTrimmable {
                    entry.trim(width - keep)
                }
            } else {
                guard let simple = entry as? KVCacheSimple else {
                    preconditionFailure(
                        "tree rollback needs a cache that can keep chosen rows, got \(type(of: entry))"
                    )
                }
                simple.keepSpeculativeRows(keepRows, width: width)
            }
            continue
        }
        // A recurrent layer without a capture cannot be rebuilt, and leaving it
        // holding a rejected branch would silently corrupt every later token.
        precondition(
            captureIndex < captures.count,
            "rollback got \(captures.count) gated-delta captures for more recurrent caches")
        let capture = captures[captureIndex]
        captureIndex += 1

        recurrent[0] = capture.convState(rows: keepRows)
        recurrent[1] = capture.recurrentState(rows: keepRows)
        recurrent.advance(-(width - keep))
    }
}
