//! Deterministic hashing. Per-tensor digests plus a manifest hash that is
//! stable across re-containerization: it depends only on tensor identity and
//! content, never on the filename or archive packing.

use blake3::Hasher;

/// One tensor's identity for manifest purposes.
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Hex blake3 digest of the tensor's raw bytes.
    pub digest: String,
}

pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Format a digest as it appears in reports: `blake3:<hex>`.
pub fn tagged(hex: &str) -> String {
    format!("blake3:{hex}")
}

/// Compute the manifest hash from a set of tensor entries.
///
/// Entries are sorted by name and fed into the hasher with explicit length
/// prefixes so the encoding is canonical and unambiguous. Renaming the file or
/// repacking the archive does not change this value.
pub fn manifest_hash(entries: &mut [TensorEntry]) -> String {
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut h = Hasher::new();
    h.update(b"assay-manifest-v1\n");
    for e in entries.iter() {
        update_field(&mut h, e.name.as_bytes());
        update_field(&mut h, e.dtype.as_bytes());
        // shape as length-prefixed sequence of u64 LE
        h.update(&(e.shape.len() as u64).to_le_bytes());
        for d in &e.shape {
            h.update(&d.to_le_bytes());
        }
        update_field(&mut h, e.digest.as_bytes());
    }
    tagged(h.finalize().to_hex().as_ref())
}

fn update_field(h: &mut Hasher, field: &[u8]) {
    h.update(&(field.len() as u64).to_le_bytes());
    h.update(field);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, dtype: &str, shape: &[u64], digest: &str) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            digest: digest.to_string(),
        }
    }

    fn sample() -> Vec<TensorEntry> {
        vec![
            entry("a.weight", "F32", &[2, 3], "aa"),
            entry("b.weight", "F32", &[4], "bb"),
        ]
    }

    #[test]
    fn manifest_is_tagged_and_stable() {
        let mut e = sample();
        let h1 = manifest_hash(&mut e);
        assert!(h1.starts_with("blake3:"));
        let mut again = sample();
        assert_eq!(h1, manifest_hash(&mut again));
    }

    #[test]
    fn manifest_is_independent_of_input_order() {
        let mut a = sample();
        let mut b: Vec<TensorEntry> = sample().into_iter().rev().collect();
        assert_eq!(manifest_hash(&mut a), manifest_hash(&mut b));
    }

    #[test]
    fn manifest_changes_with_any_tensor_field() {
        let base = manifest_hash(&mut sample());

        let mut renamed = sample();
        renamed[0].name = "z.weight".into();
        assert_ne!(base, manifest_hash(&mut renamed), "tensor name must matter");

        let mut retyped = sample();
        retyped[0].dtype = "BF16".into();
        assert_ne!(base, manifest_hash(&mut retyped), "dtype must matter");

        let mut reshaped = sample();
        reshaped[0].shape = vec![3, 2];
        assert_ne!(base, manifest_hash(&mut reshaped), "shape must matter");

        let mut recontent = sample();
        recontent[0].digest = "ab".into();
        assert_ne!(base, manifest_hash(&mut recontent), "content must matter");
    }

    #[test]
    fn length_prefixing_makes_the_encoding_unambiguous() {
        // Without length prefixes, ("ab","c") and ("a","bc") would collide.
        let mut x = vec![entry("ab", "c", &[], "")];
        let mut y = vec![entry("a", "bc", &[], "")];
        assert_ne!(manifest_hash(&mut x), manifest_hash(&mut y));
    }

    #[test]
    fn adding_a_tensor_changes_the_manifest() {
        let base = manifest_hash(&mut sample());
        let mut more = sample();
        more.push(entry("c.weight", "F32", &[1], "cc"));
        assert_ne!(base, manifest_hash(&mut more));
    }

    #[test]
    fn tagged_and_blake3_hex_agree() {
        let hex = blake3_hex(b"payload");
        assert_eq!(tagged(&hex), format!("blake3:{hex}"));
        assert_eq!(hex.len(), 64);
    }
}
