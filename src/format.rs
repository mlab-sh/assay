//! Format detection. We sniff magic bytes first and treat the file extension
//! only as a hint — `assay` refuses to guess when the bytes disagree.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
    /// Raw pickle stream or a torch zip-container wrapping pickle(s).
    Pickle,
    Unknown,
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const ZIP_MAGIC: &[u8; 4] = b"PK\x03\x04";

/// Detect format from a leading byte sample plus the path (extension hint).
pub fn detect(path: &Path, head: &[u8]) -> Format {
    // GGUF: unambiguous magic.
    if head.len() >= 4 && &head[0..4] == GGUF_MAGIC {
        return Format::Gguf;
    }

    // Torch containers are zip archives — treat as pickle (we look inside later).
    if head.len() >= 4 && &head[0..4] == ZIP_MAGIC {
        return Format::Pickle;
    }

    // Raw pickle protocol 2+ streams start with the PROTO opcode (0x80).
    if head.first() == Some(&0x80) {
        return Format::Pickle;
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());

    match ext.as_deref() {
        // Extension is explicit; the structural check happens in the parser.
        Some("safetensors") => Format::Safetensors,
        Some("gguf") => Format::Gguf,
        Some("bin" | "pt" | "pth" | "ckpt" | "pkl" | "pickle") => Format::Pickle,
        _ => {
            if looks_like_safetensors(head) {
                Format::Safetensors
            } else {
                Format::Unknown
            }
        }
    }
}

fn looks_like_safetensors(head: &[u8]) -> bool {
    if head.len() < 9 {
        return false;
    }
    let len = u64::from_le_bytes(head[0..8].try_into().unwrap());
    // Header length must be plausible and the data must begin with a JSON object.
    len > 0 && len < (1 << 32) && head[8] == b'{'
}

/// Extensions/names that we should attempt to scan when walking a directory.
pub fn is_candidate_artifact(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.is_empty() {
        return false;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    matches!(
        ext.as_deref(),
        Some("safetensors" | "gguf" | "bin" | "pt" | "pth" | "ckpt" | "pkl" | "pickle")
    )
}
