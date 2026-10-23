/// The set of token ids a constraint allows from one state, sorted ascending.
/// May include the vocabulary's EOS id (exactly when the state is accepting).
public struct TokenMask: Sendable, Hashable {
    public let allowed: [Int]
    public var count: Int { allowed.count }
    public var isEmpty: Bool { allowed.isEmpty }
}

/// Computes masks for one (constraint, vocabulary) pair.
struct MaskComputer<Constraint: DecodingConstraint>: Sendable {
    let constraint: Constraint
    let vocabulary: Vocabulary
    let trie: VocabularyTrie

    init(constraint: Constraint, vocabulary: Vocabulary) {
        self.constraint = constraint
        self.vocabulary = vocabulary
        self.trie = VocabularyTrie(vocabulary)
    }

    /// Trie walk with subtree pruning. Returns the mask and how many
    /// `advance` calls it took (the cost the mask cache exists to avoid).
    func compute(_ state: Constraint.State) -> (mask: TokenMask, advances: Int) {
        var allowed: [Int] = []
        var advances = 0
        var pending: [(node: Int, state: Constraint.State)] = [(0, state)]
        while let item = pending.popLast() {
            let nodeIndex = item.node
            let current = item.state
            guard trie.nodes.indices.contains(nodeIndex) else { continue }
            for edge in trie.nodes[nodeIndex].edges {
                advances += 1
                guard let next = constraint.advance(current, byte: edge.byte),
                      trie.nodes.indices.contains(edge.child) else { continue }
                if let token = trie.nodes[edge.child].token { allowed.append(token) }
                pending.append((edge.child, next))
            }
        }
        if constraint.isAccepting(state) { allowed.append(vocabulary.eosID) }
        allowed.sort()
        return (TokenMask(allowed: allowed), advances)
    }

    /// The obvious O(vocabulary × token length) algorithm. Kept as the test
    /// oracle for ``compute(_:)``: two independent algorithms agreeing on every
    /// reachable state is the evidence the trie pruning is sound.
    func computeNaive(_ state: Constraint.State) -> TokenMask { computeNaiveCounted(state).mask }

    func computeNaiveCounted(_ state: Constraint.State) -> (mask: TokenMask, advances: Int) {
        var allowed: [Int] = []
        var advances = 0
        for (id, bytes) in vocabulary.tokens.enumerated() {
            var current: Constraint.State? = state
            for byte in bytes {
                guard let state = current else { break }
                advances += 1
                current = constraint.advance(state, byte: byte)
            }
            if current != nil { allowed.append(id) }
        }
        if constraint.isAccepting(state) { allowed.append(vocabulary.eosID) }
        return (TokenMask(allowed: allowed), advances)
    }
}

/// A fixed-capacity LRU map with O(1) lookup, insert and eviction (hash map
/// plus an index-linked list over a slot array, so no per-node allocation).
///
/// Capacity `0` disables caching. Growth is bounded by `capacity` no matter how
/// many distinct states a long decode visits.
struct LRUCache<Key: Hashable, Value> {
    private struct Slot {
        var key: Key
        var value: Value
        var newer: Int?
        var older: Int?
    }

    let capacity: Int
    private var slots: [Slot] = []
    private var index: [Key: Int] = [:]
    private var newest: Int?
    private var oldest: Int?
    private(set) var evictions = 0

    init(capacity: Int) {
        self.capacity = max(0, capacity)
    }

    var count: Int { index.count }

    mutating func value(for key: Key) -> Value? {
        guard let slot = index[key], slots.indices.contains(slot) else { return nil }
        moveToFront(slot)
        return slots[slot].value
    }

    mutating func insert(_ value: Value, for key: Key) {
        guard capacity > 0 else { return }
        if let slot = index[key], slots.indices.contains(slot) {
            slots[slot].value = value
            moveToFront(slot)
            return
        }
        let slot: Int
        if slots.count < capacity {
            slots.append(Slot(key: key, value: value, newer: nil, older: nil))
            slot = slots.count - 1
        } else {
            guard let victim = oldest, slots.indices.contains(victim) else { return }
            unlink(victim)
            index[slots[victim].key] = nil
            evictions += 1
            slots[victim] = Slot(key: key, value: value, newer: nil, older: nil)
            slot = victim
        }
        index[key] = slot
        linkAtFront(slot)
    }

    /// Keys from most to least recently used (test hook).
    var keysByRecency: [Key] {
        var keys: [Key] = []
        var cursor = newest
        while let slot = cursor, slots.indices.contains(slot), keys.count < slots.count {
            keys.append(slots[slot].key)
            cursor = slots[slot].older
        }
        return keys
    }

    private mutating func moveToFront(_ slot: Int) {
        guard newest != slot else { return }
        unlink(slot)
        linkAtFront(slot)
    }

    private mutating func unlink(_ slot: Int) {
        let newer = slots[slot].newer
        let older = slots[slot].older
        if let newer { slots[newer].older = older } else { newest = older }
        if let older { slots[older].newer = newer } else { oldest = newer }
        slots[slot].newer = nil
        slots[slot].older = nil
    }

    private mutating func linkAtFront(_ slot: Int) {
        slots[slot].older = newest
        slots[slot].newer = nil
        if let newest { slots[newest].newer = slot }
        newest = slot
        if oldest == nil { oldest = slot }
    }
}
