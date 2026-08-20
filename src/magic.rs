//! Recognizing what a run of bytes announces itself as.
//!
//! Used wherever `assay` finds bytes that nothing accounts for: a hole between
//! tensors, a tail after the last one, data appended to a pickle. A file
//! signature there is not proof of anything on its own, but it is the
//! difference between "some bytes" and "an ELF executable".

/// File signatures worth naming when they turn up in bytes nothing claims.
const MAGICS: &[(&[u8], &str)] = &[
    (b"\x7fELF", "an ELF executable"),
    (b"\xfe\xed\xfa\xce", "a Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "a Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "a Mach-O universal binary"),
    (b"PK\x03\x04", "a ZIP archive"),
    (b"\x1f\x8b", "a gzip stream"),
    (b"BZh", "a bzip2 stream"),
    (b"\xfd7zXZ", "an xz stream"),
    (b"7z\xbc\xaf\x27\x1c", "a 7-Zip archive"),
    (b"%PDF-", "a PDF document"),
    (b"\x89PNG", "a PNG image"),
    (b"\x80\x02", "a python pickle stream"),
    (b"\x80\x03", "a python pickle stream"),
    (b"\x80\x04", "a python pickle stream"),
    (b"\x80\x05", "a python pickle stream"),
    (b"#!/", "a script shebang"),
    (b"MZ", "a DOS/PE executable"),
];

/// Name the file format a byte run announces itself as, if any.
pub fn identify(region: &[u8]) -> Option<&'static str> {
    MAGICS
        .iter()
        .find(|(magic, _)| region.starts_with(magic))
        .map(|(_, label)| *label)
}

/// A short, escaped preview of bytes we are about to report on.
pub fn preview(region: &[u8]) -> String {
    let mut out = String::new();
    for b in region.iter().take(32) {
        match b {
            0x20..=0x7e => out.push(*b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    if region.len() > 32 {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_signatures_are_named() {
        assert_eq!(identify(b"\x7fELF\x02"), Some("an ELF executable"));
        assert_eq!(identify(b"PK\x03\x04"), Some("a ZIP archive"));
        assert_eq!(identify(b"\x80\x04\x95"), Some("a python pickle stream"));
        assert_eq!(identify(b"%PDF-1.7"), Some("a PDF document"));
    }

    #[test]
    fn ordinary_bytes_are_not_named() {
        assert_eq!(identify(b"just text"), None);
        assert_eq!(identify(b""), None);
        assert_eq!(identify(&[0u8; 8]), None);
    }

    #[test]
    fn preview_escapes_and_truncates() {
        assert_eq!(preview(b"abc"), "abc");
        assert_eq!(preview(b"a\x00b"), "a\\x00b");
        assert!(preview(&[0x41u8; 64]).ends_with('…'));
        assert_eq!(preview(&[0x41u8; 64]).chars().count(), 33);
    }
}
