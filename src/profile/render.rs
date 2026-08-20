//! Rendering for layer profiles and drift profiles: a terminal sparkline and a
//! faithful 1D SVG line chart. The SVG is deliberately 1D, a per-layer line,
//! not a deforming 2D→3D projection, so it can't imply structure that isn't
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
        styler.red(&format!(
            "anomalous layers: {}",
            anomalous_labels.join(", ")
        ))
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
    sparkline_values(
        "layer profile",
        &values,
        &anomalous,
        &labels,
        metric,
        styler,
    )
}

/// A standalone SVG line/area chart. `anomalous` lists (index, magnitude) to mark.
pub fn svg_values(title: &str, values: &[f64], anomalous: &[(usize, f64)]) -> String {
    let w = 900.0;
    let h = 320.0;
    let pad = 40.0;
    let n = values.len().max(1);
    let min = values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .min(0.0);
    let max = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-9);
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
        line.push_str(&format!(
            "{}{:.2},{:.2}",
            if i == 0 { "" } else { " " },
            x(i),
            y(*v)
        ));
    }
    let mut area = line.clone();
    area.push_str(&format!(
        " {:.2},{:.2} {:.2},{:.2}",
        x(n - 1),
        y(min),
        x(0),
        y(min)
    ));

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
        &format!("assay layer profile: {metric} ({} layers)", points.len()),
        &values,
        &anomalous,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Anomaly;

    fn plain() -> Styler {
        Styler::new(false)
    }

    fn point(layer: u64, l2: f64, anomalous: bool) -> ProfilePoint {
        ProfilePoint {
            layer,
            l2,
            mean_kurtosis: l2 / 2.0,
            max_abs: l2 * 2.0,
            params: 100,
            sparsity: 0.0,
            anomaly: anomalous.then(|| Anomaly {
                metric: "l2".into(),
                mads: 9.5,
                severity: "medium".into(),
            }),
        }
    }

    #[test]
    fn bar_has_one_block_per_value_and_spans_the_range() {
        let s = bar(&[0.0, 0.5, 1.0], &[false, false, false], &plain());
        assert_eq!(s.chars().count(), 3);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], BLOCKS[0], "min maps to the lowest block");
        assert_eq!(
            chars[2],
            BLOCKS[BLOCKS.len() - 1],
            "max maps to the highest"
        );
        assert!(chars[0] < chars[1] && chars[1] < chars[2], "monotonic");
    }

    #[test]
    fn a_flat_profile_does_not_fake_a_shape() {
        // Equal values must not be stretched into a fake peak.
        let s = bar(&[7.0, 7.0, 7.0], &[false; 3], &plain());
        assert!(s.chars().all(|c| c == BLOCKS[0]), "{s}");
    }

    #[test]
    fn bar_is_empty_for_no_values() {
        assert_eq!(bar(&[], &[], &plain()), "");
    }

    #[test]
    fn anomalous_blocks_are_colorized_only_when_color_is_on() {
        let colored = bar(&[1.0, 2.0], &[false, true], &Styler::new(true));
        assert!(colored.contains("\x1b[31m"), "anomaly should be red");
        let plain_out = bar(&[1.0, 2.0], &[false, true], &plain());
        assert!(!plain_out.contains('\x1b'));
    }

    #[test]
    fn sparkline_reports_count_metric_range_and_anomalies() {
        let points = vec![point(0, 1.0, false), point(1, 4.0, true)];
        let out = sparkline(&points, "l2", &plain());
        assert!(out.contains("layer profile"));
        assert!(out.contains("(2 layers, metric=l2)"));
        assert!(out.contains("min=1.0000  max=4.0000"));
        assert!(out.contains("anomalous layers: 1"));
    }

    #[test]
    fn sparkline_says_so_when_nothing_is_anomalous() {
        let points = vec![point(0, 1.0, false), point(1, 2.0, false)];
        let out = sparkline(&points, "l2", &plain());
        assert!(out.contains("no anomalous layers"));
    }

    #[test]
    fn sparkline_handles_an_empty_profile() {
        assert!(sparkline(&[], "l2", &plain()).contains("no layers"));
    }

    #[test]
    fn sparkline_selects_the_requested_metric() {
        let points = vec![point(0, 1.0, false), point(1, 4.0, false)];
        assert!(sparkline(&points, "max_abs", &plain()).contains("max=8.0000"));
        assert!(sparkline(&points, "mean_kurtosis", &plain()).contains("max=2.0000"));
    }

    #[test]
    fn svg_is_well_formed_and_carries_one_point_per_value() {
        let out = svg_values("drift", &[0.0, 1.0, 0.5], &[]);
        assert!(out.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(out.trim_end().ends_with("</svg>"));
        assert!(out.contains("drift"));
        let line = out
            .split("<polyline points=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("polyline");
        assert_eq!(line.split_whitespace().count(), 3);
    }

    #[test]
    fn svg_marks_anomalies_with_a_labelled_dot() {
        let out = svg_values("drift", &[0.1, 0.9], &[(1, 12.0)]);
        assert_eq!(out.matches("<circle").count(), 1);
        assert!(out.contains("index 1 (12.0)"));
    }

    #[test]
    fn svg_of_a_single_layer_does_not_divide_by_zero() {
        let out = svg_values("one", &[1.0], &[]);
        assert!(out.contains("<polyline"));
        assert!(!out.contains("NaN"), "{out}");
    }

    #[test]
    fn svg_layer_profile_titles_itself_with_the_metric() {
        let out = svg(&[point(0, 1.0, false), point(1, 2.0, true)], "l2");
        assert!(out.contains("assay layer profile: l2 (2 layers)"));
        assert_eq!(out.matches("<circle").count(), 1);
    }
}
