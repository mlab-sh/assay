# Changelog

All notable changes to `assay` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Byte accounting now covers GGUF as well as safetensors.** The invariant is
  the same: every byte of the file belongs to the header, to a declared tensor,
  or to the alignment padding the format requires, and anything else is a
  storage channel nobody reads. Tensor sizes are computed from the ggml type
  and shape, which the parser previously read and discarded.

  It comes with a safety valve, because the size table is knowledge about ggml
  rather than something the file states. The layouts of `Q8_1` and the `IQ`
  family have moved between ggml versions, so they are marked unknown instead
  of guessed. And the table is cross-checked against the file: if the computed
  sizes contradict the declared offsets anywhere, the report emits
  `GGUF_ACCOUNTING_INCOMPLETE` (info) and reports no unaccounted bytes at all,
  rather than blaming a file for a table that might be wrong.

- **The secret scanner knows the credentials that actually turn up in model
  repos.** It covered AWS, GitHub, Slack, `sk-` and Google, but not `hf_`, the
  Hugging Face user access token, which is the single most likely secret to be
  committed to a Hugging Face repo. Added: `hf_` and `api_org_` tokens,
  Anthropic keys (`sk-ant-`, matched before the generic `sk-` rule so they are
  reported at high confidence), GitLab, Docker Hub, the remaining GitHub and
  Slack prefixes, temporary AWS keys (`ASIA`), and Google service account
  credentials, which are recognized by their JSON shape rather than a prefix.

  The list of sibling files was also stuck on the old repo layout. It now reads
  `chat_template.jinja` and `chat_template.json`, where Hugging Face puts chat
  templates today, plus `adapter_config.json`, `preprocessor_config.json` and
  `processor_config.json`.

- **The repo's executable code is now scanned, not just its tensor
  containers.** A Hugging Face repo can ship `modeling_foo.py` and an
  `auto_map` entry in `config.json`; `from_pretrained(trust_remote_code=True)`
  imports that module and runs everything at its top level before reading a
  single weight. It is the most direct execution path in the current ecosystem
  and `assay` did not look at it at all.

  Every `.py` shipped alongside a model is now an artifact with its own verdict
  and file hash, and the config files are parsed to find which of them a loader
  would execute (`auto_map` in `config.json`, `tokenizer_config.json`,
  preprocessor and processor configs, plus `custom_pipelines`, in both the
  string and the two-element list form). New findings:
  `REMOTE_CODE_AUTO_MAP` (this file is wired to a loader entry point),
  `REMOTE_CODE_EXTERNAL` (the mapping points at another repo, so loading pulls
  code this repo does not contain), `REMOTE_CODE_UNRESOLVED`,
  `REMOTE_CODE_PRESENT`, `PY_DANGEROUS_CALL` and `PY_OBFUSCATION`.

  Dangerous constructs are reported with their line and scored by when they
  run: `high` at module level, because that executes on import, `medium` inside
  a function body. Matching is call-aware, so `model.eval()` and
  `torch.compile()` are not mistaken for `eval(` and `compile(`. Symlinks
  pointing out of the repo are not followed. A repo with no custom code
  produces no extra reports.

- **A whole-file digest, `hashes.file`, next to the manifest hash.** The
  manifest hash covers tensor identity and content only, which is what makes it
  survive renaming and repacking, but it also means two different files can
  present the same one: staple an archive after the last tensor and the model
  identity is unchanged. Every artifact now also carries `blake3` over its
  exact bytes, so the report can distinguish them. It is computed for every
  format, including pickle and unrecognized artifacts, which had no pinnable
  hash at all until now. Pin `manifest` to answer "is this the same model", pin
  `file` to answer "is this the same file".

### Fixed

- **A GGUF file with an out-of-bounds tensor offset was reported as CLEAN.**
  `GGUF_OFFSET_OOB` set the verdict to `untrusted` inside the parser, and the
  caller then overwrote it with `clean` on the way out, so a high-severity
  structural finding sat next to a clean verdict. The verdict the parser
  reached is now kept.

- **`PICKLE_TRUNCATED` no longer fires on every ordinary checkpoint.** A legacy
  `torch.save` file is not one pickle: it is five concatenated streams (magic,
  protocol version, sys_info, the module, the storage keys) followed by the raw
  tensor storages. The scanner read the file as a single opcode stream, ignored
  `STOP`, and died on the first storage byte, so `models/gpt2/pytorch_model.bin`
  and every other legacy checkpoint came back "analysis may be incomplete". A
  warning that fires on everything is a warning nobody reads, and it hid the
  cases where analysis really was incomplete.

  `STOP` now ends a stream, the container is split into its streams, and the
  report says exactly what was read: `PICKLE_TORCH_LEGACY` (info) names the
  number of streams analyzed in full, the byte range they occupy, and the size
  of the storage payload that carries no opcodes. Evidence is attributed per
  stream, so the report tells you *which* of the five pickles carried the
  payload. `PICKLE_TRUNCATED` now means what it says, and comes with the offset
  where analysis stopped. Bytes stapled after the last stream are reported as
  `PICKLE_TRAILING_DATA`, at `high` when they begin with a known file
  signature.

- **An empty path no longer reports an internal error.** `NO_ARTIFACTS` was an
  `info` finding carrying a `Verdict::Error`, so pointing `assay` at a path with
  nothing to scan produced exit `3` and the line `internal error`, blaming the
  scanner for a situation in which nothing had failed.

  There is now a distinct `empty` verdict and a distinct exit code `4`, with the
  summary line reading `nothing scanned`. It stays non-zero on purpose: a scan
  that verified nothing is not a pass, and a mistyped path silently turning a
  gate green is the one failure mode a gate must never have. `--allow-empty`
  brings it back to `0` for pipelines that scan an optional directory, and the
  finding is still reported either way.

  A path that does not exist at all is a different thing again, and now says so:
  `PATH_NOT_FOUND` at `high`, exit `3`, unaffected by `--allow-empty`.

- **A symlink cycle no longer multiplies the scan.** `walk()` recursed through
  `p.is_dir()`, which follows symlinks, with no canonicalization and no
  deduplication. A `sub/back -> ..` link made the same model be scanned once per
  directory level until the system's ELOOP limit stopped it: 40 full rereads of
  a 40 GB file, and a report full of duplicates. Directory recursion now goes
  through a set of canonical paths already visited, so a cycle terminates at the
  first repeat.

  Symlinked *files* are still followed, deliberately: a Hugging Face cache
  snapshot is nothing but links into `blobs/`, so refusing them would mean
  scanning nothing there. Files are deduplicated by canonical path instead, so
  two names for one inode are read once and reported once, with the other paths
  listed as `ALIASED_ARTIFACT` (info) rather than silently dropped. The repo
  code scanner walks the same way, which also fixes it missing an `auto_map` in
  a cache snapshot, where `config.json` itself is a symlink.

- **A payload hidden between two tensors no longer scans clean.** Validating
  that every `data_offsets` range is in bounds, ordered, and non-overlapping
  said nothing about the bytes *between* those ranges. A file with a 64-byte
  hole carrying an ELF header and a shell command reported `CLEAN` and exit `0`,
  including under `--deep`. `safetensors` now accounts for every byte of the
  data segment: any run no tensor claims is reported as
  `ST_UNREFERENCED_BYTES`, with absolute file offsets and an escaped preview.
  Severity is `high` when the bytes carry a recognizable file signature (ELF,
  Mach-O, PE, ZIP, gzip, bzip2, xz, 7-Zip, PDF, PNG, a pickle stream, a
  shebang), `medium` for other non-zero content, and `low` for zero-filled
  padding. Non-zero unclaimed bytes make the artifact `untrusted`.

  The same accounting closes the sibling case: anything appended after the last
  tensor, which is how one file becomes a valid `safetensors` and a valid ZIP at
  the same time. Real writers pack tensors contiguously (verified on gpt2 and
  DialoGPT: zero unclaimed bytes), so the check is silent on well-formed models.

  Note that the manifest hash still covers only tensor identity and content, by
  design. Two files with the same manifest hash can differ in unclaimed bytes,
  which is exactly why those bytes are now reported.

## [0.1.0] - 2026-08-20

First release. `assay` reads a model artifact and answers two questions without
ever loading, importing, or executing it: is this file what it claims to be, and
does loading it put the machine at risk.

### Added: commands

| Command | What it does |
|---|---|
| `scan PATH` | Identify every artifact under a path, validate its container, hash it, check its signature, and report a verdict per artifact. |
| `verify PATH` | The same checks, aimed at the provenance answer, with `--bundle` and `--key` for explicit signature material. |
| `compare SUBJECT BASELINE` | Differential weight analysis against a known-good reference of the same architecture. |

### Added: container and provenance checks

- **Format detection** from magic bytes, with the extension as a hint only. A
  GGUF file named `.bin` is GGUF; `assay` refuses to guess when the bytes and
  the name disagree.
- **Pickle code-execution scanning**: every pickle artifact is untrusted by
  default, with an opcode-level scan for `GLOBAL`, `STACK_GLOBAL`, `REDUCE`, and
  imports of `os`, `subprocess`, `builtins` and friends. Torch zip containers
  are opened and scanned inside. When a clean `safetensors` file sits in the
  same repo, the report says so.
- **safetensors structural validation**: `u64` length prefix and JSON header,
  every `data_offsets` range checked for bounds, ordering, and overlap, and
  dtype and shape declarations checked against the byte ranges they claim.
- **GGUF sanity**: magic, version, tensor count, KV metadata block, and every
  tensor offset. Embedded Jinja2 chat templates are surfaced for human review
  instead of being trusted silently.
- **Deterministic hashing**: per-tensor digests plus a manifest hash that
  survives renaming and repacking, because it depends only on tensor identity
  and content. Length-prefixed canonical encoding, so no field boundary is
  ambiguous.
- **Signature verification**: detached ed25519 signatures over the manifest
  hash, and OpenSSF model-transparency manifests. Sidecars next to the artifact
  are auto-detected. A Sigstore or cosign bundle is reported as present but
  unverified, never as trusted; `signed` is only ever printed on a real
  cryptographic pass.

### Added: weight inspection (`--deep`)

Signals with a severity, never verdicts. Tensors are read cold and streamed, so
peak RAM stays well under model size.

- Per-tensor statistics in one streaming pass: NaN and Inf integrity, mean, std,
  L2 and RMS, excess kurtosis, sparsity, and 6 sigma outlier mass.
- Per-layer profile with robust median and MAD anomaly detection, a terminal
  sparkline (`--profile`), and a faithful 1D SVG chart (`--svg`).
- Secret and string scanning over metadata, GGUF KV blocks, and sibling config
  and tokenizer files: API keys, PEM private key blocks, suspicious URLs.
  High-entropy tensor-region scanning is available behind
  `--scan-tensor-entropy` and clearly labeled experimental.
- Architectural fingerprint from naming scheme, layer count, hidden dimension,
  head count, and vocabulary size, which catches a model that claims to be one
  architecture but is structurally another.

### Added: differential analysis (`compare`)

- Matched tensor pairs streamed in lockstep, with both files mapped and neither
  held fully in RAM.
- Name canonicalization across serialization conventions (a `transformer.`
  wrapper prefix is normalized away), and weight tying recognized as a
  convention rather than reported as a divergence.
- Cross-architecture comparison refused by default (`ARCH_MISMATCH`, override
  with `--force`), because drift between different architectures means nothing.
- `STRUCTURAL_DIVERGENCE` for added, removed, or reshaped tensors;
  `LAYER_DRIFT_OUTLIER` and `TENSOR_DRIFT` for concentrated drift, scored by
  magnitude; `IDENTICAL` when drift is zero everywhere.

### Added: output and CI

- Human-readable and JSON reports, with the JSON field names treated as a
  documented contract.
- Real-time progress on stderr, auto-disabled off a TTY or with `--no-progress`.
- `--color auto|always|never`.
- Exit codes: `0` clean, `1` findings at or above `--fail-on`, `2` malformed
  artifact, `3` internal error. Worst outcome wins.

### Known limits

- K-quant and IQ GGUF tensors are reported as `STATS_DEFERRED_QUANTIZED`
  (structural information only) rather than decoded. Legacy quants and the float
  types are dequantized for real statistics.
- Payloads that only activate after quantization are not detected. The
  dequantization, streaming, and lockstep drift machinery is in place; the check
  is not.
- Sigstore and cosign chains are not verified, only reported.
- Behavioral or gradient-based fingerprinting is permanently out of scope: it
  would require executing the model.
