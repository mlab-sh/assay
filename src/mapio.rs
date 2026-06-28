//! Shared file access. Artifacts are memory-mapped (not heap-read) so peak RAM
//! stays well under model size — important for `scan --deep` and `compare`,
//! where a baseline could be 70B.

use std::fs::File;
use std::ops::Deref;
use std::path::Path;

use memmap2::Mmap;

/// A file's bytes, memory-mapped when possible (falls back to a heap read).
pub enum FileBytes {
    Mapped(Mmap),
    Heap(Vec<u8>),
}

impl Deref for FileBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            FileBytes::Mapped(m) => m,
            FileBytes::Heap(v) => v,
        }
    }
}

pub fn map_file(path: &Path) -> std::io::Result<FileBytes> {
    let f = File::open(path)?;
    let len = f.metadata()?.len();
    if len == 0 {
        return Ok(FileBytes::Heap(Vec::new()));
    }
    // SAFETY: we treat the mapping as immutable; if the file is mutated
    // concurrently results are undefined — acceptable for a one-shot scan.
    match unsafe { Mmap::map(&f) } {
        Ok(m) => Ok(FileBytes::Mapped(m)),
        Err(_) => std::fs::read(path).map(FileBytes::Heap),
    }
}
