//! Per-format analysis. Each submodule takes the raw artifact bytes and fills
//! in an `ArtifactReport` (findings, verdict, hashes).

pub mod gguf;
pub mod pickle;
pub mod safetensors;
