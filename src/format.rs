//! Format detection. We sniff magic bytes first and treat the file extension
//! only as a hint; `assay` refuses to guess when the bytes disagree.

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

    // Torch containers are zip archives, so treat them as pickle (we look inside later).
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn st_head(header_len: u64) -> Vec<u8> {
        let mut v = header_len.to_le_bytes().to_vec();
        v.push(b'{');
        v
    }

    #[test]
    fn magic_beats_extension() {
        // A GGUF file misnamed `.bin` is GGUF: bytes win over the extension.
        assert_eq!(detect(&p("model.bin"), b"GGUF\x03\0\0\0"), Format::Gguf);
        // A pickle stream misnamed `.safetensors` is still pickle.
        assert_eq!(
            detect(&p("model.safetensors"), b"\x80\x02X"),
            Format::Pickle
        );
    }

    #[test]
    fn zip_container_is_pickle() {
        assert_eq!(
            detect(&p("pytorch_model.bin"), b"PK\x03\x04rest"),
            Format::Pickle
        );
    }

    #[test]
    fn extension_is_the_fallback_hint() {
        // Bytes say nothing; the extension decides, and the parser checks later.
        assert_eq!(
            detect(&p("model.safetensors"), b"\0\0\0\0"),
            Format::Safetensors
        );
        assert_eq!(detect(&p("model.gguf"), b"\0\0\0\0"), Format::Gguf);
        for ext in ["bin", "pt", "pth", "ckpt", "pkl", "pickle"] {
            assert_eq!(
                detect(&p(&format!("model.{ext}")), b"\0\0\0\0"),
                Format::Pickle,
                "extension {ext}"
            );
        }
    }

    #[test]
    fn headerless_safetensors_shape_is_recognized() {
        // No extension at all, but the u64 length prefix + `{` is unambiguous.
        assert_eq!(detect(&p("weights"), &st_head(120)), Format::Safetensors);
        // Implausible header length is not enough to claim safetensors.
        assert_eq!(detect(&p("weights"), &st_head(0)), Format::Unknown);
        assert_eq!(detect(&p("weights"), &st_head(1 << 40)), Format::Unknown);
    }

    #[test]
    fn refuses_to_guess() {
        assert_eq!(detect(&p("README.md"), b"# hello"), Format::Unknown);
        assert_eq!(detect(&p("noext"), b""), Format::Unknown);
    }

    #[test]
    fn candidate_artifacts_are_model_files_only() {
        for name in [
            "model.safetensors",
            "model.gguf",
            "pytorch_model.bin",
            "w.pt",
            "w.pth",
            "w.ckpt",
            "w.pkl",
            "w.pickle",
        ] {
            assert!(is_candidate_artifact(&p(name)), "{name} should be scanned");
        }
        for name in [
            "README.md",
            "config.json",
            "tokenizer.json",
            "vocab.txt",
            "model.onnx",
            "tf_model.h5",
            "noext",
        ] {
            assert!(!is_candidate_artifact(&p(name)), "{name} should be skipped");
        }
    }

    #[test]
    fn candidate_extension_match_is_case_insensitive() {
        assert!(is_candidate_artifact(&p("MODEL.SAFETENSORS")));
        assert!(is_candidate_artifact(&p("Model.GGUF")));
    }
}
