//! Numeric dtype model + cold byte → f64 value streaming.
//!
//! Phase 2 reads tensor data **cold** — it reinterprets raw bytes as numbers
//! without ever loading a framework. This module knows the element layout of
//! every non-quantized dtype we support (both safetensors and GGUF) and yields
//! values one at a time so callers can accumulate stats in O(1) memory.
//!
//! Quantized GGUF block formats live in [`crate::dequant`]; this module only
//! handles formats where the raw bytes *are* the (encoded) scalars.

/// A unified element type spanning safetensors and GGUF non-quantized dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    /// Byte size of one element.
    pub fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Parse a safetensors dtype string.
    pub fn from_safetensors(s: &str) -> Option<DType> {
        Some(match s {
            "F64" => DType::F64,
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I64" => DType::I64,
            "I32" => DType::I32,
            "I16" => DType::I16,
            "I8" => DType::I8,
            "U8" => DType::U8,
            "BOOL" => DType::Bool,
            _ => return None,
        })
    }

}

/// IEEE-754 half-precision → f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let val = match exp {
        0 => {
            // subnormal / zero
            (frac as f32) * 2f32.powi(-24)
        }
        0x1f => {
            if frac == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => {
            let e = exp as i32 - 15;
            (1.0 + frac as f32 / 1024.0) * 2f32.powi(e)
        }
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// bfloat16 → f32 (just the high 16 bits of an f32).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Number of whole elements of `dtype` that fit in `raw`.
pub fn element_count(raw: &[u8], dtype: DType) -> usize {
    raw.len() / dtype.size()
}

/// Decode element `i` of a non-quantized tensor as f64.
pub fn decode_index(raw: &[u8], dtype: DType, i: usize) -> f64 {
    let off = i * dtype.size();
    match dtype {
        DType::F64 => f64::from_le_bytes(raw[off..off + 8].try_into().unwrap()),
        DType::F32 => f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as f64,
        DType::F16 => f16_to_f32(u16::from_le_bytes(raw[off..off + 2].try_into().unwrap())) as f64,
        DType::BF16 => bf16_to_f32(u16::from_le_bytes(raw[off..off + 2].try_into().unwrap())) as f64,
        DType::I64 => i64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) as f64,
        DType::I32 => i32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as f64,
        DType::I16 => i16::from_le_bytes(raw[off..off + 2].try_into().unwrap()) as f64,
        DType::I8 => (raw[off] as i8) as f64,
        DType::U8 => raw[off] as f64,
        DType::Bool => (raw[off] != 0) as u8 as f64,
    }
}

/// Stream every element of a non-quantized tensor as f64, calling `f` per value.
///
/// `raw` must be the tensor's contiguous byte range. Trailing bytes that don't
/// complete an element are ignored. This is deliberately re-runnable so callers
/// can do a second pass (e.g. outlier counting against a now-known std) without
/// materializing the tensor.
pub fn stream_values(raw: &[u8], dtype: DType, mut f: impl FnMut(f64)) {
    let n = element_count(raw, dtype);
    for i in 0..n {
        f(decode_index(raw, dtype, i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_roundtrip_basic() {
        // 1.0 in f16 is 0x3C00.
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -2.0 is 0xC000.
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert!(f16_to_f32(0x7C01).is_nan());
        assert!(f16_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn bf16_basic() {
        // 1.0f32 = 0x3F800000 -> high 16 bits 0x3F80.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
    }

    #[test]
    fn stream_f32() {
        let mut bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut sum = 0.0;
        stream_values(&bytes, DType::F32, |x| sum += x);
        assert_eq!(sum, 6.0);
    }
}
