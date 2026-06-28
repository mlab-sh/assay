# Try assay in two minutes

You don't need a lab. Pull a couple of real models off Hugging Face, point
`assay` at them, and watch it tell clean weights from untrusted ones. Every
example below uses small, public models that download in seconds.

```sh
cargo install assay        # or grab a static binary from releases
pip install -U huggingface_hub   # for the `hf` downloader (curl works too)
```

---

## 1. The clean / dirty contrast — `gpt2`

`gpt2` is the perfect first target: the same repo ships **both** a modern
`safetensors` file **and** a legacy pickle (`pytorch_model.bin`). One is safe by
construction, the other can run code when you load it. `assay` should call it.

```sh
hf download gpt2 --local-dir ./gpt2
assay scan ./gpt2/
```

**What you should see:** `model.safetensors` comes back clean, while
`pytorch_model.bin` is flagged `untrusted` for pickle/RCE risk — plus an info
note that a safe alternative exists in the same repo. That single line is the
whole pitch: *you were one `torch.load` away from running someone else's code,
and you had a clean file sitting right next to it.*

---

## 2. A clean modern model — safetensors only

```sh
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./minilm
assay scan ./minilm/ --json
```

A well-behaved repo. You should get exit code `0` and a manifest hash you can
pin in CI. This is what "boring and trustworthy" looks like.

---

## 3. A GGUF model — the local-inference format

The format everyone actually runs on their laptop via llama.cpp / Ollama.

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir ./qwen-gguf
assay scan ./qwen-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

`assay` validates the GGUF structure and offsets, and — importantly — surfaces
any **embedded chat template** for review instead of silently trusting it.
GGUF can't execute code, but a Jinja template smuggled in metadata is still a
surface worth a human glance.

---

## 4. Self-test: prove the pickle scanner actually fires

Like the EICAR file for antivirus, you should be able to verify detection on
demand with a **harmless** fixture. This pickle runs a benign `echo` — nothing
destructive — and exists only to confirm `assay` flags it.

```sh
python3 - <<'PY'
import pickle, os
class Probe:
    def __reduce__(self):
        return (os.system, ("echo assay-detection-selftest",))
with open("selftest.pkl", "wb") as f:
    pickle.dump(Probe(), f)
print("wrote selftest.pkl (benign echo payload)")
PY

assay scan ./selftest.pkl --fail-on high ; echo "exit=$?"
```

**What you should see:** a `PICKLE_RCE_RISK` finding pointing at the
`os.system` reduce, and a non-zero exit. If you get exit `0` here, the scanner
is broken — that's the point of the self-test. Delete it after:

```sh
rm selftest.pkl
```

---

## 5. Drop it into CI

```sh
# fail the pipeline if any model artifact is untrusted
assay scan ./models/ --json --fail-on high \
  | tee assay-report.json
```

Non-zero exit blocks the merge. The JSON report is your audit trail: format,
verdict, findings, and a stable manifest hash per artifact.

---

## What this does *not* prove yet

Phase 1 is provenance & integrity. It catches the lazy attacker: pickle
payloads, malformed containers, unsigned weights, identity drift. It does **not**
yet detect a backdoor hidden in the weights, and it does **not** yet catch a
payload that only wakes up after GGUF quantization (that's Phase 3, and it's
hard on purpose). `assay` will always tell you the confidence of a finding — it
never claims to catch what it can't.

Found a model `assay` should have flagged and didn't? That's the best possible
bug report. Open an issue with the repo id.