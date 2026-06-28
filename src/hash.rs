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
