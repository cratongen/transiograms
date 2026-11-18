use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use varvite_core::codes::check_codes;
use varvite_core::{Code, Direction, Grid, MISSING, Mat, Result, Run, SequenceSet, invalid};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experimental {
    /// Nominal lag `l * Delta` (or the given lags).
    pub lags: Vec<f64>,
    /// Mean `|h|` of the pairs (NaN when none).
    #[serde(with = "varvite_core::serde_mat::values")]
    pub mean_lags: Vec<f64>,
    /// `F_ij` per lag; lengths for the downhole estimator.
    #[serde(with = "varvite_core::serde_mat::vec")]
    pub counts: Vec<Mat>,
    /// `F_ij / sum_k F_ik`; NaN row when the tail total is 0.
    #[serde(with = "varvite_core::serde_mat::vec")]
    pub probabilities: Vec<Mat>,
    pub tail_totals: Vec<Vec<f64>>,
    pub directed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairFilter {
    All,
    SameHole,
    DifferentHole,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairSpec {
    /// (azimuth, dip) degrees; None = omnidirectional.
    pub direction: Option<[f64; 2]>,
    pub lag: f64,
    pub n_lags: usize,
    /// Default `lag / 2`.
    pub lag_tolerance: f64,
    /// Degrees, default 22.5.
    pub angle_tolerance: f64,
    /// Horizontal, default inf.
    pub bandwidth: f64,
    /// Default inf.
    pub vertical_bandwidth: f64,
    /// Default true; false counts both orientations.
    pub directed: bool,
    /// Default All.
    pub pairs: PairFilter,
}

const BLOCK: usize = 1024;

fn from_counts(
    lags: Vec<f64>,
    mean_lags: Vec<f64>,
    counts: Vec<Mat>,
    directed: bool,
) -> Experimental {
    let tail_totals: Vec<Vec<f64>> = counts
        .iter()
        .map(|c| c.row_iter().map(|r| r.sum()).collect())
        .collect();
    let probabilities = counts
        .iter()
        .zip(&tail_totals)
        .map(|(c, t)| {
            Mat::from_fn(c.nrows(), c.ncols(), |i, j| {
                if t[i] > 0.0 {
                    c[(i, j)] / t[i]
                } else {
                    f64::NAN
                }
            })
        })
        .collect();
    Experimental {
        lags,
        mean_lags,
        counts,
        probabilities,
        tail_totals,
        directed,
    }
}

struct Cone {
    u: [f64; 3],
    sin_a: f64,
    cos_a: f64,
    sin_d: f64,
    cos_d: f64,
    cos_tol: f64,
    bandwidth: f64,
    vertical_bandwidth: f64,
}

impl Cone {
    fn new(direction: [f64; 2], spec: &PairSpec) -> Self {
        let [a, d] = direction.map(f64::to_radians);
        Self {
            u: [a.sin() * d.cos(), a.cos() * d.cos(), -d.sin()],
            sin_a: a.sin(),
            cos_a: a.cos(),
            sin_d: d.sin(),
            cos_d: d.cos(),
            cos_tol: spec.angle_tolerance.to_radians().cos(),
            bandwidth: spec.bandwidth,
            vertical_bandwidth: spec.vertical_bandwidth,
        }
    }

    fn accepts(&self, s: [f64; 3], length: f64) -> bool {
        let along = s[0] * self.u[0] + s[1] * self.u[1] + s[2] * self.u[2];
        if along <= 0.0 || along < self.cos_tol * length {
            return false;
        }
        let horizontal = (s[0] * self.cos_a - s[1] * self.sin_a).abs();
        let azimuthal = s[0] * self.sin_a + s[1] * self.cos_a;
        let vertical = (s[2] * self.cos_d + azimuthal * self.sin_d).abs();
        horizontal <= self.bandwidth && vertical <= self.vertical_bandwidth
    }
}

fn check_spec(spec: &PairSpec) -> Result<()> {
    if !(spec.lag > 0.0 && spec.lag.is_finite()) {
        return invalid(format!("lag must be positive and finite, got {}", spec.lag));
    }
    if spec.n_lags == 0 {
        return invalid("n_lags must be at least 1");
    }
    if !(spec.lag_tolerance > 0.0 && spec.lag_tolerance.is_finite()) {
        return invalid(format!(
            "lag_tolerance must be positive and finite, got {}",
            spec.lag_tolerance
        ));
    }
    if !(spec.angle_tolerance > 0.0 && spec.angle_tolerance <= 90.0) {
        return invalid(format!(
            "angle_tolerance must be in (0, 90] degrees, got {}",
            spec.angle_tolerance
        ));
    }
    if spec.bandwidth.is_nan() || spec.bandwidth < 0.0 {
        return invalid(format!("bandwidth must be >= 0, got {}", spec.bandwidth));
    }
    if spec.vertical_bandwidth.is_nan() || spec.vertical_bandwidth < 0.0 {
        return invalid(format!(
            "vertical_bandwidth must be >= 0, got {}",
            spec.vertical_bandwidth
        ));
    }
    if let Some(d) = spec.direction
        && d.iter().any(|v| !v.is_finite())
    {
        return invalid(format!("direction {d:?} must be finite"));
    }
    Ok(())
}

pub fn experimental(
    coords: &[[f64; 3]],
    codes: &[Code],
    holes: Option<&[u32]>,
    k: usize,
    spec: &PairSpec,
) -> Result<Experimental> {
    let n = codes.len();
    if coords.len() != n {
        return invalid(format!("{} coordinates but {n} codes", coords.len()));
    }
    if let Some(h) = holes
        && h.len() != n
    {
        return invalid(format!("{} hole ids but {n} codes", h.len()));
    }
    if spec.pairs != PairFilter::All && holes.is_none() {
        return invalid("pairs=same_hole or different_hole needs holes");
    }
    check_codes(codes, k)?;
    check_spec(spec)?;
    let active: Vec<usize> = (0..n).filter(|&i| codes[i] != MISSING).collect();
    if let Some(&i) = active
        .iter()
        .find(|&&i| coords[i].iter().any(|v| !v.is_finite()))
    {
        return invalid(format!(
            "row {i} has non-finite coordinates {:?}",
            coords[i]
        ));
    }
    let cone = spec.direction.map(|d| Cone::new(d, spec));
    let directed = spec.directed && cone.is_some();
    let (lag, tol, n_lags) = (spec.lag, spec.lag_tolerance, spec.n_lags);
    let reach = n_lags as f64 * lag + tol;
    let cell = |x: [f64; 3]| x.map(|v| (v / reach).floor() as i64);
    let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for &i in &active {
        buckets.entry(cell(coords[i])).or_default().push(i);
    }
    let blocks: Vec<(Vec<Mat>, Vec<f64>)> = active
        .par_chunks(BLOCK)
        .map(|tails| {
            let mut counts = vec![Mat::zeros(k, k); n_lags];
            let mut distance = vec![0.0; n_lags];
            for &t in tails {
                let c = cell(coords[t]);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let Some(heads) = buckets.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                            else {
                                continue;
                            };
                            for &h in heads {
                                if h == t {
                                    continue;
                                }
                                if let Some(ids) = holes {
                                    let same = ids[h] == ids[t];
                                    match spec.pairs {
                                        PairFilter::SameHole if !same => continue,
                                        PairFilter::DifferentHole if same => continue,
                                        _ => {}
                                    }
                                }
                                let s: [f64; 3] =
                                    std::array::from_fn(|a| coords[h][a] - coords[t][a]);
                                let d = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
                                if let Some(cone) = &cone
                                    && !cone.accepts(s, d)
                                    && (directed || !cone.accepts(s.map(|v| -v), d))
                                {
                                    continue;
                                }
                                let lo = ((d - tol) / lag).ceil().max(1.0) as usize;
                                let hi = ((d + tol) / lag).floor().min(n_lags as f64);
                                if hi < 1.0 {
                                    continue;
                                }
                                for l in lo..=hi as usize {
                                    if (d - l as f64 * lag).abs() <= tol {
                                        counts[l - 1][(codes[t] as usize, codes[h] as usize)] +=
                                            1.0;
                                        distance[l - 1] += d;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (counts, distance)
        })
        .collect();
    let mut counts = vec![Mat::zeros(k, k); n_lags];
    let mut distance = vec![0.0; n_lags];
    for (c, d) in blocks {
        for l in 0..n_lags {
            counts[l] += &c[l];
            distance[l] += d[l];
        }
    }
    let mean_lags = (0..n_lags)
        .map(|l| {
            let pairs = counts[l].sum();
            if pairs > 0.0 {
                distance[l] / pairs
            } else {
                f64::NAN
            }
        })
        .collect();
    let lags = (1..=n_lags).map(|l| l as f64 * lag).collect();
    Ok(from_counts(lags, mean_lags, counts, directed))
}

/// Exact overlap lengths `|[a0, a1] ∩ [b0 - h, b1 - h] ∩ [r0, r1 - h]|` by a two-pointer sweep.
fn run_lengths(run: &Run, h: f64, k: usize) -> Mat {
    let n = run.codes.len();
    let r0 = run.start[0];
    let r1 = run.end[n - 1];
    let mut m = Mat::zeros(k, k);
    let mut first = 0;
    for a in 0..n {
        let a0 = run.start[a].max(r0);
        let a1 = run.end[a].min(r1 - h);
        if a1 <= a0 {
            continue;
        }
        while first < n && run.end[first] - h <= a0 {
            first += 1;
        }
        let mut b = first;
        while b < n && run.start[b] - h < a1 {
            let lo = a0.max(run.start[b] - h);
            let hi = a1.min(run.end[b] - h);
            if hi > lo {
                m[(run.codes[a] as usize, run.codes[b] as usize)] += hi - lo;
            }
            b += 1;
        }
    }
    m
}

pub fn downhole(
    seqs: &SequenceSet,
    direction: Direction,
    lags: &[f64],
    gap: f64,
) -> Result<Experimental> {
    if lags.is_empty() {
        return invalid("give at least one lag");
    }
    if let Some(h) = lags.iter().find(|h| !(**h > 0.0 && h.is_finite())) {
        return invalid(format!("lags must be positive and finite, got {h}"));
    }
    if gap.is_nan() || gap < 0.0 {
        return invalid(format!("gap must be >= 0, got {gap}"));
    }
    let k = seqs.n_categories;
    let runs = seqs.runs(direction, gap);
    let per_run: Vec<Vec<Mat>> = runs
        .par_iter()
        .map(|run| lags.iter().map(|&h| run_lengths(run, h, k)).collect())
        .collect();
    let mut counts = vec![Mat::zeros(k, k); lags.len()];
    for run in per_run {
        for (total, m) in counts.iter_mut().zip(run) {
            *total += m;
        }
    }
    Ok(from_counts(lags.to_vec(), lags.to_vec(), counts, true))
}

pub fn on_grid(
    codes: &[Code],
    grid: &Grid,
    mask: Option<&[bool]>,
    offset: [i64; 3],
    n_lags: usize,
    k: usize,
) -> Result<Experimental> {
    let n = grid.n_cells();
    if codes.len() != n {
        return invalid(format!("{} codes for {n} cells", codes.len()));
    }
    if let Some(m) = mask
        && m.len() != n
    {
        return invalid(format!("mask has {} values for {n} cells", m.len()));
    }
    if offset == [0; 3] {
        return invalid("offset must be non-zero");
    }
    if n_lags == 0 {
        return invalid("n_lags must be at least 1");
    }
    check_codes(codes, k)?;
    let active = |c: usize| codes[c] != MISSING && mask.is_none_or(|m| m[c]);
    let counts: Vec<Mat> = (1..=n_lags as i64)
        .into_par_iter()
        .map(|l| {
            let mut m = Mat::zeros(k, k);
            let Some(d) = offset
                .iter()
                .map(|v| v.checked_mul(l))
                .collect::<Option<Vec<i64>>>()
            else {
                return m;
            };
            let d = [d[0], d[1], d[2]];
            for c in (0..n).filter(|&c| active(c)) {
                if let Some(h) = grid.offset(c, d)
                    && active(h)
                {
                    m[(codes[c] as usize, codes[h] as usize)] += 1.0;
                }
            }
            m
        })
        .collect();
    let step = grid.lag(offset).iter().map(|v| v * v).sum::<f64>().sqrt();
    let lags: Vec<f64> = (1..=n_lags).map(|l| l as f64 * step).collect();
    let mean_lags = lags
        .iter()
        .zip(&counts)
        .map(|(&h, c)| if c.sum() > 0.0 { h } else { f64::NAN })
        .collect();
    Ok(from_counts(lags, mean_lags, counts, true))
}
