//! Rendering for layer profiles and drift profiles: a terminal sparkline and a
//! faithful 1D SVG line chart. The SVG is deliberately 1D — a per-layer line,
//! not a deforming 2D→3D projection — so it can't imply structure that isn't
//! there. The core renderers take plain value arrays so both the Phase 2 layer
//! profile and the `compare` drift profile reuse them.

use super::ProfilePoint;
use crate::style::Styler;

const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// The block-character bar for `values`; entries flagged in `anomalous` are
/// colorized.
pub fn bar(values: &[f64], anomalous: &[bool], styler: &Styler) -> String {
    if values.is_empty() {
        return String::new();
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).abs();
    let mut out = String::new();
    for (i, v) in values.iter().enumerate() {
        let idx = if span <= f64::EPSILON {
            0
        } else {
            (((v - min) / span) * (BLOCKS.len() - 1) as f64).round() as usize
        };
        let ch = BLOCKS[idx.min(BLOCKS.len() - 1)].to_string();
        if anomalous.get(i).copied().unwrap_or(false) {
            out.push_str(&styler.red(&ch));
        } else {
            out.push_str(&ch);
        }
    }
    out
}

/// Generic sparkline block with a title line and min/max + anomaly footer.
pub fn sparkline_values(
    title: &str,
    values: &[f64],
    anomalous: &[bool],
    anomalous_labels: &[String],
    metric: &str,
    styler: &Styler,
) -> String {
    if values.is_empty() {
        return format!("{} (no layers)", styler.bold(title));
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let footer = if anomalous_labels.is_empty() {
        styler.dim("no anomalous layers")
    } else {
        styler.red(&format!("anomalous layers: {}", anomalous_labels.join(", ")))
    };
    format!(
        "{} {} ({} layers, metric={})\n  {}\n  {}",
        styler.bold(title),
        bar(values, anomalous, styler),
        values.len(),
        metric,
        styler.dim(&format!("min={min:.4}  max={max:.4}")),
        footer,
    )
}

/// Layer-profile sparkline (Phase 2).
pub fn sparkline(points: &[ProfilePoint], metric: &str, styler: &Styler) -> String {
    let values: Vec<f64> = points
        .iter()
        .map(|p| match metric {
            "mean_kurtosis" => p.mean_kurtosis,
            "max_abs" => p.max_abs,
            _ => p.l2,
        })
        .collect();
    let anomalous: Vec<bool> = points.iter().map(|p| p.anomaly.is_some()).collect();
    let labels: Vec<String> = points
        .iter()
        .filter(|p| p.anomaly.is_some())
        .map(|p| p.layer.to_string())
        .collect();
    sparkline_values("layer profile", &values, &anomalous, &labels, metric, styler)
}

/// A standalone SVG line/area chart. `anomalous` lists (index, magnitude) to mark.
pub fn svg_values(title: &str, values: &[f64], anomalous: &[(usize, f64)]) -> String {
    let w = 900.0;
    let h = 320.0;
    let pad = 40.0;
    let n = values.len().max(1);
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-9);
    let span = (max - min).max(1e-9);

    let x = |i: usize| -> f64 {
        if n == 1 {
            pad
        } else {
            pad + (w - 2.0 * pad) * (i as f64) / ((n - 1) as f64)
        }
    };
    let y = |v: f64| -> f64 { h - pad - (h - 2.0 * pad) * (v - min) / span };

    let mut line = String::new();
    for (i, v) in values.iter().enumerate() {
        line.push_str(&format!("{}{:.2},{:.2}", if i == 0 { "" } else { " " }, x(i), y(*v)));
    }
    let mut area = line.clone();
    area.push_str(&format!(" {:.2},{:.2} {:.2},{:.2}", x(n - 1), y(min), x(0), y(min)));

    let mut dots = String::new();
    for (i, mag) in anomalous {
        dots.push_str(&format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"5\" fill=\"#e5484d\"><title>index {} ({:.1})</title></circle>",
            x(*i),
            y(values[*i]),
            i,
            mag
        ));
    }

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" font-family=\"sans-serif\">\n\
         <rect width=\"{w}\" height=\"{h}\" fill=\"#0b0d12\"/>\n\
         <text x=\"{pad}\" y=\"24\" fill=\"#e6e6e6\" font-size=\"15\">{title}</text>\n\
         <polygon points=\"{area}\" fill=\"#3b82f6\" fill-opacity=\"0.18\"/>\n\
         <polyline points=\"{line}\" fill=\"none\" stroke=\"#3b82f6\" stroke-width=\"2\"/>\n\
         {dots}\n\
         <text x=\"{pad}\" y=\"{ty}\" fill=\"#8a8f98\" font-size=\"11\">min {min:.4}  max {max:.4}</text>\n\
         </svg>\n",
        ty = h - 12.0,
    )
}

/// Layer-profile SVG (Phase 2).
pub fn svg(points: &[ProfilePoint], metric: &str) -> String {
    let values: Vec<f64> = points
        .iter()
        .map(|p| match metric {
            "mean_kurtosis" => p.mean_kurtosis,
            "max_abs" => p.max_abs,
            _ => p.l2,
        })
        .collect();
    let anomalous: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.anomaly.as_ref().map(|a| (i, a.mads)))
        .collect();
    svg_values(
        &format!("assay layer profile — {metric} ({} layers)", points.len()),
        &values,
        &anomalous,
    )
}
