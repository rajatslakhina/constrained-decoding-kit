# ConstrainedDecoding

**Your on-device model gets JSON right most of the time. Your decoder has to accept every output. This library closes that gap by construction rather than by retry.**

ConstrainedDecoding is a grammar-constrained decoding engine in Swift. You give it a schema, a tokenizer vocabulary and anything that produces next-token logits. Before each sampling step it removes every token that would break the schema. With the default budget, every decode ends in end-of-sequence with a document that is valid for the schema, whatever the model prefers.

```
Same model, same seed, same sampler, two decoder instances. Only the mask differs.

Unconstrained:  Sure! Here's the ticket as JSON. I'd call it "high" priority, ...
                {"title": "Crash when rotating the photo editor", "priority": "P1",
                 "estimate_hours": 6.5, "blocking": "yes", "labels": ["crash", "editor", "ipad",],}
                Let me know if you need anything else!
                → validator: expected { at $ (byte 0)

Constrained:    {"title":"Crash when rotating the photo editor","priority":"high",
                 "estimate_hours":6,"blocking":true,"labels":["crash","editor","ipad"]}
                → valid
```

Both outputs are pinned byte-for-byte by the test suite (`testGreedyGoldenOutput`, `testUnconstrainedBaselineEmitsInvalidOutput`). In a 20-seed sweep at temperature 1.4, **0/20** unconstrained outputs were schema-valid and **20/20** constrained outputs were (`testDocumentedSeedSweep`).

Demo app: (added after the companion repo is pushed — see below)

---

## Why this matters

Structured output is how an LLM feature talks to the rest of an app. A triage screen, an intent router or a form filler needs a value it can decode, not prose. There are three ways to get one:

| Approach | What you get | What it costs |
|---|---|---|
| **Prompt for JSON, parse, retry** | Easy to start | No guarantee. Every retry adds latency and cost, and a 0.5% failure rate becomes a crash or a support ticket at fleet scale. |
| **Use a grammar engine someone else maintains**: Apple's Foundation Models guided generation (`@Generable`) for the system model; llama.cpp's GBNF / JSON-schema grammars; XGrammar or llguidance (C++/Rust engines that also run on device, inside MLC-LLM, WebLLM and llama.cpp) | Battle-tested and fast | Each one is tied to its runtime or needs a native bridge into your Swift stack. You inherit its schema subset, its performance profile and its upgrade cadence. |
| **Own a small engine in Swift** (this repo) | One contract for any runtime that exposes logits, written in the language your team reviews | You own the correctness argument, the performance budget and the failure modes. |

This is a build-vs-buy decision. The lead's question is not "can we get JSON?". It is "which guarantees do we need, and which component owns them?". This repo is the third row, kept small enough to read in one sitting, with every guarantee written down and a test behind each one. It also shows what you need to understand before you choose the second row with your eyes open.

## The guarantees, and how each is enforced

| Guarantee | Mechanism | Evidence |
|---|---|---|
| **Validity**: every output that ends in EOS satisfies the schema | Per-step token mask from a pushdown matcher over the schema | `testConstrainedOutputIsAlwaysSchemaValid`: 4 samplers × 15 seeds, each output checked by an **independent** validator |
| **Termination**: the decode ends, even if the model never wants to stop | The schema is bounded, so the language is finite and no document exceeds `maxEncodedBytes` bytes. Compilation refuses schemas above 1 MiB (`SchemaLimits.maxEncodedBytes`), so the step budget also bounds memory. Every token is at least one byte, so with the default budget of `maxEncodedBytes + 1` steps (`DecodingConfiguration.maxTokens = nil`) only EOS can remain at the end. The decoder enforces that budget itself and never exceeds `DecodingConfiguration.absoluteMaxTokens` (1 MiB + 1). A wrong or saturated bound from a custom constraint still cannot loop forever or grow memory without limit (`testBudgetHasAnAbsoluteCeiling`). A smaller caller cap trades the EOS guarantee for a latency ceiling and reports `.tokenBudgetExhausted`. | `testTerminatesUnderAnAdversarialModel` (a model that loves whitespace, long tokens and gives EOS `-inf`); `testDefaultBudgetIsTheConstraintBound` (`maxLength: 4096`); `testDecoderStepCapHoldsEvenWhenTheConstraintBoundIsWrong` |
| **No dead ends**: no state where nothing is allowed | The matcher is live: every state it reaches can still complete. An init-time check confirms the vocabulary can spell every byte the grammar may need. | `testEveryReachableStateIsLiveAndBounded` (random walks); `testVocabularyMissingAStructuralByteIsRejectedAtInit`; `testUnderdeclaredCoverageSurfacesAsDeadEnd` (a typed error, not a trap) |
| **Reproducibility**: same seed, same output | SplitMix64 PRNG; ties go to the lowest token id | The golden output and the sweep numbers are asserted by tests that run on Linux and on macOS in CI. Bit-exactness *across* platforms is not promised, because `exp`/`log` come from the platform's libm. |
| **Shared-cache safety**: concurrent decodes on one decoder | Actor isolation. Per-decode state lives in locals, and the cache holds only pure values. | `testConcurrentDecodesOnASharedDecoderMatchSequentialRuns`: 8 decodes that provably overlap (the test asserts more than one is in flight at once), an 8-entry cache that is forced to evict, and byte-for-byte equality with fresh sequential runs |

### Tests built to fail against a broken implementation

A test that still passes when the implementation is broken only looks like coverage. Each load-bearing property has a control that plants a bug and asserts the check catches it:

- **Validator**: `LeakyConstraint` adds a hole that lets `#` through. The broken constraint accepts its own output. The independent `SchemaValidator` rejects it.
- **Liveness checker**: `TrapdoorConstraint` adds a byte that leads to a state that is neither accepting nor extendable. The random-walk checker must report it.
- **Step cap**: the same leaky constraint plus a model that wants `#` forever. Without the decoder's cap the decode never returns; with it, the test requires exactly `maxOutputBytes + 1` steps.
- **Trie mask**: compared with a naive per-token scan on every state of several random walks. The test also requires more than 10 distinct masks, so two algorithms that agreed on a constant would fail.
- **Cache key**: within a single constraint, the states after `["x` and after `["y","x` have the same top frame. `",` is legal in the first and illegal in the second (`maxItems: 2`). A cache keyed on the top frame would hand one state the other's mask, and the decoder's re-advance check would throw.
- **Concurrency**: the provider yields inside every call, and the test fails if the decodes never overlapped.
- **Cancellation**: the provider never suspends or checks cancellation, so only the decoder's own `Task.checkCancellation()` can stop it. With that line deleted, the test fails.

## Architecture

```
               ┌──────────────── ConstrainedDecoder (actor) ────────────────┐
 Schema ─────► │ CompiledSchema ─► JSONSchemaConstraint (pushdown matcher)   │
               │                        │ advance(state, byte) -> state?     │
 Vocabulary ─► │ VocabularyTrie ─► MaskComputer ─► LRUCache<State, Mask>     │
               │                        │ allowed ids (+EOS)                 │
 LogitsProvider ── logits ──────► Sampler (mask, renormalise, pick)          │
               └──────────────────────────────────────────────────────────────┘
 Real model:  ModelBackend (model ids, model-width logits)
              ─► MappedModel + TokenizerMapping ─► LogitsProvider + Vocabulary
```

| Type | Responsibility |
|---|---|
| `Schema` / `CompiledSchema` | A bounded JSON Schema subset, validated and flattened into a node table capped at `SchemaLimits.maxNodes`. Computes `maxEncodedBytes` with saturating arithmetic. |
| `DecodingConstraint` | The protocol seam: `initialState`, a pure `advance(state, byte)`, `isAccepting`, `maxOutputBytes` and coverage requirements. States are `Hashable` values. |
| `JSONSchemaConstraint` / `MatcherState` | The pushdown matcher. Its state is the **whole** frame stack. |
| `UnconstrainedText` | The baseline constraint. The demo runs the same engine with it, so the mask is the only variable. |
| `Vocabulary` / `VocabularyTrie` | Token bytes with validation (no empty or duplicate tokens). The byte trie lets one rejected byte prune every token that shares that prefix. |
| `TokenizerMapping` / `MappedModel` / `ModelBackend` | Adapts a real tokenizer: specials and empty pieces are never sampled, mid-vocabulary and multiple EOS ids are handled, and duplicate spellings (byte-fallback pieces) merge by log-sum-exp. Context ids are translated back to model ids. |
| `MaskComputer`, `LRUCache` | Trie-walk masks, a naive oracle for tests, and an O(1), fixed-capacity LRU keyed by matcher state. |
| `Sampler`, `SplitMix64` | Greedy or temperature sampling over the allowed set only. Handles NaN and ±inf. Reports **masked mass**: how much of the sampled distribution the mask removed. |
| `SchemaValidator` | An independent recursive-descent JSON parser and schema checker. It shares no code with the matcher. |
| `ReferenceTextModel` | A deterministic stand-in "model": an n-gram copy model over one reference text. It makes the demo and its numbers reproducible. |
| `ConstrainedDecodingUI` | A SwiftUI playground with two lanes, a masked-mass strip and a seed sweep. It compiles only where SwiftUI exists. |

## Design decisions, trade-offs and rejected alternatives

**1. Canonical compact JSON only.** The matcher accepts one encoding per value: no insignificant whitespace, and properties in declaration order. *Trade-off:* the model cannot format freely or reorder keys. *Why:* with free whitespace the language is infinite. A model that likes spaces can emit them forever, and the termination argument is lost. *Rejected:* whitespace with a cap. It adds a counter to every state and multiplies the distinct states the cache has to hold.

**2. A bounded schema subset.** Every string has `maxLength`, every integer `maxDigits` and every array `maxItems`. All properties are required. There are no floats, no escapes in strings and no non-ASCII. Compiled size is capped at 4,096 nodes, and the longest describable document at 1 MiB. *Trade-off:* expressiveness. *Why:* boundedness turns termination into arithmetic, and the caps keep that arithmetic small. 200 properties × 200 × 200 is a few lines of `Schema` but 8 million nodes. Three nested nodes, `[[String(4096)] × 1024] × 1024`, describe 4 GB of output. Both are refused at compile time (`testWideNestedSchemaIsRejectedBeforeItExplodes`, `testEncodedSizeCapBoundsSmallButHugeSchemas`). *Rejected:* optional properties and `additionalProperties`. Each turns a deterministic matcher into a set of parallel states.

**3. A per-state pushdown matcher plus a memoised mask, not ahead-of-time token automata.** XGrammar-style engines precompute most masks per grammar to reach server-level throughput. *Trade-off:* a cache miss here costs a trie walk. Over random walks through the ticket schema the trie made about 40% fewer `advance` calls than a naive per-token scan (88,988 against 149,645 in one test run), and on no state did it make more (`testTriePruningNeverCostsMoreThanTheNaiveScan`). Inside a string almost every token is legal, so no walk is cheap there. That case is what the cache is for: a second identical decode makes zero `advance` calls (`testMaskCacheIsReusedAcrossDecodes`). *Rejected for this scope:* grammar precompilation. It is the right next step if profiling on a device shows mask time competing with the forward pass.

**4. The cache key is the full stack, not the top frame.** The counterexample is above. *Trade-off:* more distinct keys. *Why:* a top-frame key would be faster and wrong.

**5. A forced-token fast path, and its honest limit.** When exactly one token is allowed, the model is not called (`testForcedStepsDoNotCallTheModel`). In the golden decode that covers **7 of 41 steps** (34 model calls), mostly `,`, `:` and the closing `]`, `}` and EOS. Property names are *not* forced: a BPE vocabulary offers several spellings of `"priority` (`"`, `"priority`, ...), so the model is still consulted on every key. *Next step, deliberately not taken:* jump-forward decoding. It appends the whole forced literal and re-tokenises, at the cost of non-canonical token boundaries and more complexity in the context the model sees.

**6. Validity is not correctness, and the engine measures the gap.** Masking renormalises the model's distribution over what the schema allows. Each step records **masked mass**, the probability the mask removed from the distribution being sampled. In the golden decode, exactly four of 41 steps exceed 99%, and each one is a place where the model wanted something illegal. Step 0 is `{"` where the model wanted to start with prose. Then `high"` where it wanted `"P1"`, the `,` after `6` where it wanted `.5`, and `true` where it wanted `"yes"`. Across the free-text title (`Crash` through `editor`, steps 3–13) masked mass stays under 6%. `testGreedyGoldenOutput` asserts both the spike list and the title bound. Property-name steps also run high (`title` 92%, `"blocking` 95%, `"labels` 97%). That is why the demo's strip draws a 99% guide line. In production that is the metric to watch per field. For example: "masked mass on `priority` doubled after the model update".

**7. The mask can force unnatural tokenisations.** If the model's natural token was `": "` and the grammar forbids the space, the model has to spell `":` differently than it was trained to. Token healing would reduce this distortion. It is out of scope, and named here so the omission isn't mistaken for an oversight.

**8. An actor with no invariants held across `await`.** The model call is the only suspension point in a step. Matcher state, context, RNG and stats are locals of `decode`. The actor owns only the cache, whose entries are a pure function of their key. *Rejected:* a class with a lock. It would also be correct, but the actor keeps the property local and checkable.

**9. No traps on external input.** `Sources` has no force-unwraps except one `try!` with a written proof: `Vocabulary.demo(corpus:)` caps the corpus-derived tokens, so any input stays far below the vocabulary limit (`testDemoVocabularyIsBoundedForHugeCorpora`). Every index that comes from outside is bounds-checked: token ids, logits, schema node ids and tokenizer ids. Remaining subscripts index internal structures by invariant (the LRU slot links and the trie builder). Size arithmetic saturates (`Saturating.add` / `.multiply`). Wrong logit widths, empty masks, oversized schemas and non-pure constraints are typed errors (`testImpureConstraintIsATypedError` drives the last one with a constraint that starts refusing bytes mid-decode).

## Naming note

The library's error type is `ConstrainedDecodingError`, so it never shadows `Swift.DecodingError` in a Codable `catch`. `Schema` keeps its short name for readability. If a file also imports SwiftData, write `ConstrainedDecoding.Schema`.

## Connecting a real model

```swift
// 1. Describe the model's tokenizer, one entry per model id. Convert each
//    piece's display form (GPT-2 "Ġ", SentencePiece "▁", "<0x41>") to real bytes first.
let mapping = try TokenizerMapping(pieces: modelPieces.map { piece in
    if piece.isEndOfTurn { return .endOfSequence }
    if piece.isSpecial { return .special }
    return .bytes(piece.utf8Bytes)
})

// 2. Wrap your runtime's forward pass (MLX, llama.cpp, Core ML...).
struct MyBackend: ModelBackend {
    func logits(afterModelTokens context: [Int]) async throws -> [Float] { /* forward pass */ }
}

// 3. Decode.
let decoder = try ConstrainedDecoder(constraint: JSONSchemaConstraint(schema), vocabulary: mapping.vocabulary)
let result = try await decoder.decode(using: MappedModel(mapping: mapping, backend: MyBackend()))
```

`TokenizerMappingTests` covers this with a toy tokenizer that has a BOS special at id 0, an empty piece, an EOS in the middle of the vocabulary, a byte-fallback duplicate and a second end-of-turn id.

## How to run it

```bash
git clone https://github.com/rajatslakhina/constrained-decoding-kit.git
cd constrained-decoding-kit
swift test                      # macOS or Linux, Swift 6.0+
```

To see it run as an app, use the companion demo repo (link above). Or add the package to any app target:

```swift
.package(url: "https://github.com/rajatslakhina/constrained-decoding-kit.git", from: "1.0.0")
```

## Limitations (stated plainly)

- The demo "model" is an n-gram copy model over one reference text, not a neural network. It makes every number in this README reproducible and says nothing about real-model quality or latency.
- No benchmark on a real on-device model exists yet, so there are no latency or throughput claims.
- Strings are printable ASCII with no escapes. There are no floats, optional properties or `anyOf`.
- Logits arrive as `[Float]` per step. A production integration would keep them on the GPU and apply the mask there.

## Verification

- **Linux, Swift 6.0.3**: clean build (`rm -rf .build`) with `-warnings-as-errors`, 0 warnings. `swift test`: **65 tests, 0 failures**. On Linux the SwiftUI module compiles to nothing because of `#if canImport(SwiftUI)`, so this run does not cover it.
- **CI** ([Actions](../../actions)) has two jobs:
  - Linux (Swift 6.0 container): build and test with warnings as errors.
  - macOS 15: build and test, then an `xcodebuild` build of every product for `generic/platform=iOS Simulator`. That last step is what compiles `ConstrainedDecodingUI`.
- **Simulator**: the companion demo repo states exactly what was and was not run on a Simulator.

## License

MIT
