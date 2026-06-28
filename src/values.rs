//! Pull-based value iteration over a tensor, decoding cold bytes to f64 on the
//! fly. Unlike the push-based `stream_values`/`stream_quant`, an iterator lets
//! `compare` walk two tensors in lockstep (cosine / rel-L2 need element pairing)
//! while holding only one block of dequantized values at a time.

use crate::dequant::{self, GgmlClass, QuantKind, BLOCK_ELEMS};
use crate::numeric::{self, DType};

// The Quant variant carries a 32-value decode buffer; boxing it would add
// per-element indirection on a hot path, so we accept the size difference.
#[allow(clippy::large_enum_variant)]
pub enum ValueIter<'a> {
    Plain {
        raw: &'a [u8],
        dtype: DType,
        i: usize,
        n: usize,
    },
    Quant {
        kind: QuantKind,
        raw: &'a [u8],
        bs: usize,
        n_blocks: usize,
        blk: usize,
        buf: [f64; BLOCK_ELEMS],
        pos: usize,
        filled: usize,
        emitted: u64,
        n_elements: u64,
    },
}

impl<'a> ValueIter<'a> {
    /// Build an iterator for a safetensors tensor. `None` for unsupported dtype.
    pub fn safetensors(raw: &'a [u8], dtype_str: &str) -> Option<ValueIter<'a>> {
        let dtype = DType::from_safetensors(dtype_str)?;
        Some(ValueIter::Plain {
            raw,
            dtype,
            i: 0,
            n: numeric::element_count(raw, dtype),
        })
    }

    /// Build an iterator for a GGUF tensor. `None` for deferred (k-quant/IQ).
    pub fn gguf(raw: &'a [u8], type_id: u32, n_elements: u64) -> Option<ValueIter<'a>> {
        match dequant::classify(type_id).1 {
            GgmlClass::Plain(dtype) => Some(ValueIter::Plain {
                raw,
                dtype,
                i: 0,
                n: numeric::element_count(raw, dtype),
            }),
            GgmlClass::QuantLegacy(kind) => {
                let bs = kind.block_size();
                Some(ValueIter::Quant {
                    kind,
                    raw,
                    bs,
                    n_blocks: raw.len() / bs,
                    blk: 0,
                    buf: [0.0; BLOCK_ELEMS],
                    pos: 0,
                    filled: 0,
                    emitted: 0,
                    n_elements,
                })
            }
            GgmlClass::Deferred => None,
        }
    }
}

impl Iterator for ValueIter<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        match self {
            ValueIter::Plain { raw, dtype, i, n } => {
                if *i >= *n {
                    return None;
                }
                let v = numeric::decode_index(raw, *dtype, *i);
                *i += 1;
                Some(v)
            }
            ValueIter::Quant {
                kind,
                raw,
                bs,
                n_blocks,
                blk,
                buf,
                pos,
                filled,
                emitted,
                n_elements,
            } => {
                if *pos >= *filled {
                    if *blk >= *n_blocks || *emitted >= *n_elements {
                        return None;
                    }
                    dequant::decode_block(*kind, &raw[*blk * *bs..*blk * *bs + *bs], buf);
                    *blk += 1;
                    let remaining = *n_elements - *emitted;
                    *filled = (remaining as usize).min(BLOCK_ELEMS);
                    *pos = 0;
                }
                let v = buf[*pos];
                *pos += 1;
                *emitted += 1;
                Some(v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_iter_matches_stream() {
        let mut bytes = Vec::new();
        for v in [1.0f32, -2.0, 3.5, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let it = ValueIter::safetensors(&bytes, "F32").unwrap();
        let got: Vec<f64> = it.collect();
        assert_eq!(got, vec![1.0, -2.0, 3.5, 0.0]);
    }

    #[test]
    fn quant_iter_q8_0() {
        let d: u16 = 0x3800; // 0.5
        let mut raw = Vec::new();
        raw.extend_from_slice(&d.to_le_bytes());
        for j in 0..32 {
            raw.push(j as i8 as u8);
        }
        let it = ValueIter::gguf(&raw, 8 /* Q8_0 */, 32).unwrap();
        let got: Vec<f64> = it.collect();
        assert_eq!(got.len(), 32);
        assert!((got[4] - 2.0).abs() < 1e-6); // 0.5 * 4
    }

    #[test]
    fn deferred_returns_none() {
        assert!(ValueIter::gguf(&[0u8; 16], 12 /* Q4_K */, 256).is_none());
    }
}
