//! GGUF block-quantization decoders.
//!
//! GGUF quantized tensors store *encoded* weights in fixed-size blocks — the
//! raw bytes are not the weights, so computing stats on them directly would be
//! garbage. We dequantize the common **legacy** block formats (Q4_0, Q4_1,
//! Q5_0, Q5_1, Q8_0) for real per-weight stats. The **k-quant** family
//! (Q*_K) and exotic IQ types are intentionally deferred for now — their
//! decoders are subtle and shipping a wrong one would produce misleading
//! numbers. Phase 3 will build on this module to add them.
//!
//! References: ggml `dequantize_row_q*` (legacy quants, block size QK = 32).

use crate::numeric::{f16_to_f32, DType};

const QK: usize = 32;

/// Legacy quant block kinds we can fully dequantize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
}

impl QuantKind {
    /// Bytes per 32-weight block.
    pub fn block_size(self) -> usize {
        match self {
            QuantKind::Q4_0 => 2 + QK / 2,      // d(f16) + 16 nibbles
            QuantKind::Q4_1 => 2 + 2 + QK / 2,  // d + m + 16 nibbles
            QuantKind::Q5_0 => 2 + 4 + QK / 2,  // d + qh(u32) + 16 nibbles
            QuantKind::Q5_1 => 2 + 2 + 4 + QK / 2,
            QuantKind::Q8_0 => 2 + QK,          // d + 32 int8
        }
    }
}

/// Classification of a GGUF ggml tensor type.
pub enum GgmlClass {
    /// Raw bytes are scalars of this dtype (handled by `numeric`).
    Plain(DType),
    /// Block-quantized; dequantizable here.
    QuantLegacy(QuantKind),
    /// Recognized but not dequantized yet (k-quants, IQ types, Q8_1).
    Deferred,
}

/// Map a GGUF ggml type id to (display name, classification).
pub fn classify(type_id: u32) -> (&'static str, GgmlClass) {
    use GgmlClass::*;
    match type_id {
        0 => ("F32", Plain(DType::F32)),
        1 => ("F16", Plain(DType::F16)),
        2 => ("Q4_0", QuantLegacy(QuantKind::Q4_0)),
        3 => ("Q4_1", QuantLegacy(QuantKind::Q4_1)),
        6 => ("Q5_0", QuantLegacy(QuantKind::Q5_0)),
        7 => ("Q5_1", QuantLegacy(QuantKind::Q5_1)),
        8 => ("Q8_0", QuantLegacy(QuantKind::Q8_0)),
        9 => ("Q8_1", Deferred),
        10 => ("Q2_K", Deferred),
        11 => ("Q3_K", Deferred),
        12 => ("Q4_K", Deferred),
        13 => ("Q5_K", Deferred),
        14 => ("Q6_K", Deferred),
        15 => ("Q8_K", Deferred),
        16 => ("IQ2_XXS", Deferred),
        17 => ("IQ2_XS", Deferred),
        18 => ("IQ3_XXS", Deferred),
        19 => ("IQ1_S", Deferred),
        20 => ("IQ4_NL", Deferred),
        21 => ("IQ3_S", Deferred),
        22 => ("IQ2_S", Deferred),
        23 => ("IQ4_XS", Deferred),
        24 => ("I8", Plain(DType::I8)),
        25 => ("I16", Plain(DType::I16)),
        26 => ("I32", Plain(DType::I32)),
        27 => ("I64", Plain(DType::I64)),
        28 => ("F64", Plain(DType::F64)),
        30 => ("BF16", Plain(DType::BF16)),
        other => {
            // Unknown id: defer rather than guess.
            let _ = other;
            ("UNKNOWN", Deferred)
        }
    }
}

fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// Number of weights per block (always 32 for legacy quants).
pub const BLOCK_ELEMS: usize = QK;

/// Decode one block of `kind` into `out` (32 values, natural element order).
// `j` indexes the nibble array *and* drives the qh bit math, so a range loop is
// the clearest expression here.
#[allow(clippy::needless_range_loop)]
pub fn decode_block(kind: QuantKind, b: &[u8], out: &mut [f64; QK]) {
    match kind {
        QuantKind::Q8_0 => {
            let d = rd_f16(b, 0);
            for j in 0..QK {
                out[j] = (d * b[2 + j] as i8 as f32) as f64;
            }
        }
        QuantKind::Q4_0 => {
            let d = rd_f16(b, 0);
            let qs = &b[2..2 + QK / 2];
            for j in 0..QK / 2 {
                out[j] = (((qs[j] & 0x0F) as i32 - 8) as f32 * d) as f64;
                out[j + QK / 2] = (((qs[j] >> 4) as i32 - 8) as f32 * d) as f64;
            }
        }
        QuantKind::Q4_1 => {
            let d = rd_f16(b, 0);
            let m = rd_f16(b, 2);
            let qs = &b[4..4 + QK / 2];
            for j in 0..QK / 2 {
                out[j] = ((qs[j] & 0x0F) as f32 * d + m) as f64;
                out[j + QK / 2] = ((qs[j] >> 4) as f32 * d + m) as f64;
            }
        }
        QuantKind::Q5_0 => {
            let d = rd_f16(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let qs = &b[6..6 + QK / 2];
            for j in 0..QK / 2 {
                let xh0 = (((qh >> j) << 4) & 0x10) as i32;
                let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
                out[j] = ((((qs[j] & 0x0F) as i32 | xh0) - 16) as f32 * d) as f64;
                out[j + QK / 2] = ((((qs[j] >> 4) as i32 | xh1) - 16) as f32 * d) as f64;
            }
        }
        QuantKind::Q5_1 => {
            let d = rd_f16(b, 0);
            let m = rd_f16(b, 2);
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let qs = &b[8..8 + QK / 2];
            for j in 0..QK / 2 {
                let xh0 = (((qh >> j) << 4) & 0x10) as i32;
                let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
                out[j] = (((qs[j] & 0x0F) as i32 | xh0) as f32 * d + m) as f64;
                out[j + QK / 2] = (((qs[j] >> 4) as i32 | xh1) as f32 * d + m) as f64;
            }
        }
    }
}

/// Stream up to `n_elements` dequantized values from `raw`, calling `f` per
/// value. Stops at `n_elements` so block padding is not counted.
pub fn stream_quant(kind: QuantKind, raw: &[u8], n_elements: u64, mut f: impl FnMut(f64)) {
    let bs = kind.block_size();
    let n_blocks = raw.len() / bs;
    let mut emitted: u64 = 0;
    let mut buf = [0f64; QK];

    for blk in 0..n_blocks {
        if emitted >= n_elements {
            break;
        }
        decode_block(kind, &raw[blk * bs..blk * bs + bs], &mut buf);
        let take = ((n_elements - emitted) as usize).min(QK);
        for &v in &buf[..take] {
            f(v);
        }
        emitted += take as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_roundtrip() {
        // One block: scale d=0.5 (f16), qs = 0,2,4,...; value = d*q.
        let d: u16 = 0x3800; // 0.5 in f16
        let mut raw = Vec::new();
        raw.extend_from_slice(&d.to_le_bytes());
        for j in 0..QK {
            raw.push(j as i8 as u8);
        }
        let mut vals = Vec::new();
        stream_quant(QuantKind::Q8_0, &raw, QK as u64, |v| vals.push(v));
        assert_eq!(vals.len(), QK);
        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 1.0).abs() < 1e-6); // 0.5 * 2
        assert!((vals[10] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn q4_0_center_is_zero() {
        // nibble 8 -> (8-8)=0 -> 0 regardless of scale.
        let d: u16 = 0x3C00; // 1.0
        let mut raw = Vec::new();
        raw.extend_from_slice(&d.to_le_bytes());
        raw.resize(raw.len() + QK / 2, 0x88); // both nibbles = 8 -> zero
        let mut sum = 0.0;
        stream_quant(QuantKind::Q4_0, &raw, QK as u64, |v| sum += v);
        assert!(sum.abs() < 1e-6);
    }
}
