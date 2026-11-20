//! Fit-quality report: how far an ensemble of grid realizations sits from the proportions and
//! transiograms of the model it was simulated from.

use std::mem::size_of;

use serde::{Deserialize, Serialize};
use transiograms_core::codes::check_codes;
use transiograms_core::{Grid, MISSING, Mat, Realizations, Result, invalid, memory};

use crate::{TransitionModel, on_grid};

/// Standard errors below this count as this value in [`FitQuality::max_standard_errors`].
pub const SE_FLOOR: f64 = 1e-3;

/// Ensemble proportions and transiograms against the model, per realization and on average.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitQuality {
    pub k: usize,
    pub n_realizations: usize,
    #[serde(with = "transiograms_core::serde_mat::values")]
    pub target_proportions: Vec<f64>,
    /// `n_realizations * k`, realization-major; NaN for a realization with no active cell.
    #[serde(with = "transiograms_core::serde_mat::values")]
    pub proportions: Vec<f64>,
    /// Cell offsets `l * direction`, `l = 1..=n_lags`, direction-major.
    pub offsets: Vec<[i64; 3]>,
    /// Model `t_ij` at each offset.
    #[serde(with = "transiograms_core::serde_mat::vec")]
    pub model: Vec<Mat>,
    /// Mean over the realizations where row `i` has pairs; NaN when none has.
    #[serde(with = "transiograms_core::serde_mat::vec")]
    pub mean: Vec<Mat>,
    /// Spread across realizations over `sqrt(count)`; NaN with fewer than two values.
    #[serde(with = "transiograms_core::serde_mat::vec")]
    pub standard_error: Vec<Mat>,
}

impl FitQuality {
    /// Ensemble proportions, averaged over realizations with active cells.
    pub fn mean_proportions(&self) -> Vec<f64> {
        (0..self.k)
            .map(|c| {
                finite_mean((0..self.n_realizations).map(|r| self.proportions[r * self.k + c]))
            })
            .collect()
    }

    /// Largest `|mean proportion - target|`.
    pub fn proportion_error(&self) -> f64 {
        self.mean_proportions()
            .iter()
            .zip(&self.target_proportions)
            .map(|(m, t)| (m - t).abs())
            .fold(f64::NAN, f64::max)
    }

    /// Largest `|mean t_ij - model t_ij|` over every finite entry and offset.
    pub fn transiogram_error(&self) -> f64 {
        self.mean
            .iter()
            .zip(&self.model)
            .flat_map(|(m, t)| m.iter().zip(t.iter()).map(|(a, b)| (a - b).abs()))
            .fold(f64::NAN, f64::max)
    }

    /// Largest `|mean t_ij - model t_ij| / max(se, SE_FLOOR)`; NaN with one realization.
    pub fn max_standard_errors(&self) -> f64 {
        let mut z = f64::NAN;
        for ((m, t), s) in self.mean.iter().zip(&self.model).zip(&self.standard_error) {
            for ((a, b), se) in m.iter().zip(t.iter()).zip(s.iter()) {
                if se.is_nan() {
                    continue;
                }
                z = z.max((a - b).abs() / se.max(SE_FLOOR));
            }
        }
        z
    }
}

fn finite_mean(values: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for v in values.filter(|v| v.is_finite()) {
        s += v;
        n += 1;
    }
    if n == 0 { f64::NAN } else { s / n as f64 }
}

/// Compares `sims` with `model` at offsets `l * d` for each direction `d` and `l = 1..=n_lags`,
/// pooling pairs over the active cells of each realization.
pub fn fit_quality(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    sims: &Realizations,
    directions: &[[i64; 3]],
    n_lags: usize,
) -> Result<FitQuality> {
    let (k, n, cells) = (model.k(), sims.n, grid.n_cells());
    if sims.k != k {
        return invalid(format!(
            "realizations have {} categories but the model has {k}",
            sims.k
        ));
    }
    if sims.n_cells != cells || n.checked_mul(cells) != Some(sims.codes.len()) {
        return invalid(format!(
            "realizations hold {} cells each but the grid has {cells}",
            sims.n_cells
        ));
    }
    if n == 0 {
        return invalid("realizations must hold at least one realization");
    }
    if let Some(m) = mask
        && m.len() != cells
    {
        return invalid(format!("mask has {} values for {cells} cells", m.len()));
    }
    if directions.is_empty() || directions.contains(&[0; 3]) {
        return invalid("directions must be non-empty and non-zero");
    }
    if n_lags == 0 {
        return invalid("n_lags must be at least 1");
    }
    check_codes(&sims.codes, k)?;
    let m = (directions.len() as u64).saturating_mul(n_lags as u64);
    let mat = size_of::<Mat>() as u64 + 8 * (k as u64).pow(2);
    let needed = m
        .saturating_mul(mat)
        .saturating_mul(n as u64 + 4)
        .saturating_add(8 * (n as u64).saturating_mul(k as u64));
    memory::check(needed, None, "fit-quality report")?;

    let active = |c: usize| mask.is_none_or(|m| m[c]);
    let mut proportions = vec![0.0; n * k];
    for (r, codes) in sims.codes.chunks_exact(cells.max(1)).enumerate() {
        let row = &mut proportions[r * k..(r + 1) * k];
        let mut total = 0.0;
        for (c, &code) in codes.iter().enumerate() {
            if code != MISSING && active(c) {
                row[code as usize] += 1.0;
                total += 1.0;
            }
        }
        row.iter_mut().for_each(|v| *v /= total);
    }

    let mut offsets = Vec::with_capacity(m as usize);
    let mut per: Vec<Vec<Mat>> = Vec::with_capacity(m as usize);
    for &d in directions {
        offsets.extend((1..=n_lags as i64).map(|l| d.map(|v| v.saturating_mul(l))));
        let mut by_lag = vec![Vec::with_capacity(n); n_lags];
        for codes in sims.codes.chunks_exact(cells.max(1)) {
            let e = on_grid(codes, grid, mask, d, n_lags, k)?;
            for (l, p) in e.probabilities.into_iter().enumerate() {
                by_lag[l].push(p);
            }
        }
        per.extend(by_lag);
    }

    let model_t = offsets
        .iter()
        .map(|&d| model.transition(grid.lag(d)))
        .collect();
    let mut mean = Vec::with_capacity(offsets.len());
    let mut standard_error = Vec::with_capacity(offsets.len());
    for ps in &per {
        let mu = Mat::from_fn(k, k, |i, j| finite_mean(ps.iter().map(|p| p[(i, j)])));
        let se = Mat::from_fn(k, k, |i, j| {
            let v: Vec<f64> = ps
                .iter()
                .map(|p| p[(i, j)])
                .filter(|v| v.is_finite())
                .collect();
            if v.len() < 2 {
                return f64::NAN;
            }
            let c = v.len() as f64;
            let var = v.iter().map(|x| (x - mu[(i, j)]).powi(2)).sum::<f64>() / (c - 1.0);
            (var / c).sqrt()
        });
        mean.push(mu);
        standard_error.push(se);
    }

    Ok(FitQuality {
        k,
        n_realizations: n,
        target_proportions: model.proportions.clone(),
        proportions,
        offsets,
        model: model_t,
        mean,
        standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use transiograms_core::Code;

    fn model() -> TransitionModel {
        TransitionModel::from_geology(
            &[0.5, 0.3, 0.2],
            [&[12.0, 8.0, 6.0], &[12.0, 8.0, 6.0], &[2.0, 1.0, 0.8]],
            [None, None, None],
            2,
            [0.0; 3],
        )
        .unwrap()
    }

    #[test]
    fn hand_counted_two_realizations() {
        let grid = Grid::new([0.0; 3], [1.0; 3], [4, 1, 1]).unwrap();
        let sims = Realizations {
            n: 2,
            n_cells: 4,
            k: 3,
            codes: vec![0, 0, 1, 2, 1, 1, 1, MISSING],
            corrected: vec![0.0; 2],
            objective: Vec::new(),
        };
        let q = fit_quality(&model(), &grid, None, &sims, &[[1, 0, 0]], 1).unwrap();
        assert_eq!(q.proportions, vec![0.5, 0.25, 0.25, 0.0, 1.0, 0.0]);
        assert_eq!(q.mean_proportions(), vec![0.25, 0.625, 0.125]);
        assert!((q.proportion_error() - 0.325).abs() < 1e-12);
        assert_eq!(q.offsets, vec![[1, 0, 0]]);
        let (mu, se) = (&q.mean[0], &q.standard_error[0]);
        assert_eq!([mu[(0, 0)], mu[(0, 1)], mu[(0, 2)]], [0.5, 0.5, 0.0]);
        assert_eq!([mu[(1, 0)], mu[(1, 1)], mu[(1, 2)]], [0.0, 0.5, 0.5]);
        assert!(se[(0, 0)].is_nan() && mu[(2, 0)].is_nan());
        assert!((se[(1, 1)] - 0.5).abs() < 1e-12);
        let t = model().transition([1.0, 0.0, 0.0]);
        assert_eq!(q.model[0], t);
        let want = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .filter(|&(i, j)| mu[(i, j)].is_finite())
            .map(|(i, j)| (mu[(i, j)] - t[(i, j)]).abs())
            .fold(0.0, f64::max);
        assert_eq!(q.transiogram_error(), want);
    }

    #[test]
    fn rejects_mismatched_inputs() {
        let grid = Grid::new([0.0; 3], [1.0; 3], [4, 1, 1]).unwrap();
        let sims = Realizations {
            n: 1,
            n_cells: 4,
            k: 3,
            codes: vec![0, 1, 2, 0],
            corrected: vec![0.0],
            objective: Vec::new(),
        };
        let m = model();
        assert!(fit_quality(&m, &grid, None, &sims, &[[0, 0, 0]], 1).is_err());
        assert!(fit_quality(&m, &grid, None, &sims, &[[1, 0, 0]], 0).is_err());
        assert!(fit_quality(&m, &grid, Some(&[true]), &sims, &[[1, 0, 0]], 1).is_err());
        let small = Grid::new([0.0; 3], [1.0; 3], [3, 1, 1]).unwrap();
        assert!(fit_quality(&m, &small, None, &sims, &[[1, 0, 0]], 1).is_err());
        let mut bad = sims.clone();
        bad.codes[0] = 3;
        assert!(fit_quality(&m, &grid, None, &bad, &[[1, 0, 0]], 1).is_err());
        let one = fit_quality(&m, &grid, None, &sims, &[[1, 0, 0]], 1).unwrap();
        assert!(one.max_standard_errors().is_nan());
    }

    #[test]
    fn identical_on_1_and_8_threads() {
        let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], [9, 7, 6]).unwrap();
        let n_cells = grid.n_cells();
        let codes = (0..3 * n_cells)
            .map(|i| (transiograms_core::rng::splitmix(i as u64) % 3) as Code)
            .collect();
        let sims = Realizations {
            n: 3,
            n_cells,
            k: 3,
            codes,
            corrected: vec![0.0; 3],
            objective: Vec::new(),
        };
        let run = || {
            let q = fit_quality(&model(), &grid, None, &sims, &[[1, 0, 0], [0, 0, -1]], 4);
            serde_json::to_string(&q.unwrap()).unwrap()
        };
        let pool = |n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        };
        assert_eq!(pool(1).install(run), pool(8).install(run));
    }

    #[test]
    fn json_round_trip_keeps_nan() {
        let grid = Grid::new([0.0; 3], [1.0; 3], [4, 1, 1]).unwrap();
        let sims = Realizations {
            n: 2,
            n_cells: 4,
            k: 3,
            codes: vec![0, 0, 1, 2, 1, 1, 1, MISSING],
            corrected: vec![0.0; 2],
            objective: Vec::new(),
        };
        let q = fit_quality(&model(), &grid, None, &sims, &[[1, 0, 0]], 2).unwrap();
        let back: FitQuality = serde_json::from_str(&serde_json::to_string(&q).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_string(&back).unwrap(),
            serde_json::to_string(&q).unwrap()
        );
    }
}
