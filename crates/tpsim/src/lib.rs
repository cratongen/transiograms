//! Transition-probability cokriging, sequential indicator simulation and zero-temperature
//! quenching on grids (T-PROGS).

pub mod cokriging;
pub mod quench;
pub mod search;
pub mod sis;

pub use cokriging::{cokrige, correct, predict};
pub use quench::{objective, quench, quench_lags};
pub use sis::simulate;

use varvite_core::{Grid, Result, Snapped, invalid};
use varvite_transiogram::TransitionModel;

#[derive(Clone, Debug, PartialEq)]
pub struct QuenchOptions {
    pub max_iter: usize,
    pub limit: f64,
    pub tol: f64,
    pub proportion_weight: f64,
}

impl Default for QuenchOptions {
    fn default() -> Self {
        Self {
            max_iter: 3,
            limit: 0.4,
            tol: 1e-3,
            proportion_weight: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub n_realizations: usize,
    pub seed: u64,
    /// Even.
    pub max_data: usize,
    pub closeness_limit: f64,
    /// 0 = no cap.
    pub max_per_octant: usize,
    /// Relative to the largest singular value.
    pub svd_cutoff: f64,
    pub quench: Option<QuenchOptions>,
    pub memory: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            n_realizations: 1,
            seed: 0,
            max_data: 12,
            closeness_limit: 0.01,
            max_per_octant: 0,
            svd_cutoff: 1e-8,
            quench: Some(QuenchOptions::default()),
            memory: None,
        }
    }
}

/// Validates the shared inputs of `predict` and `simulate`; returns `k`.
pub(crate) fn check_inputs(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<usize> {
    let k = model.k();
    let n_cells = grid.n_cells();
    check_mask(grid, mask)?;
    if hard.cells.len() != hard.codes.len() {
        return invalid(format!(
            "hard data have {} cells and {} codes; give one code per cell",
            hard.cells.len(),
            hard.codes.len()
        ));
    }
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        if cell >= n_cells {
            return invalid(format!(
                "hard cell {cell} is outside the grid of {n_cells} cells"
            ));
        }
        if code as usize >= k {
            return invalid(format!(
                "hard code {code} at cell {cell} is not one of the model's {k} categories"
            ));
        }
        if mask.is_some_and(|m| !m[cell]) {
            return invalid(format!("hard cell {cell} lies outside the mask"));
        }
    }
    if !opts.max_data.is_multiple_of(2) {
        return invalid(format!("max_data must be even, got {}", opts.max_data));
    }
    if !(opts.closeness_limit > 0.0 && opts.closeness_limit <= 1.0) {
        return invalid(format!(
            "closeness_limit must lie in (0, 1], got {}",
            opts.closeness_limit
        ));
    }
    if !(opts.svd_cutoff.is_finite() && opts.svd_cutoff >= 0.0) {
        return invalid(format!(
            "svd_cutoff must be finite and non-negative, got {}",
            opts.svd_cutoff
        ));
    }
    if let Some(q) = &opts.quench {
        check_quench(q)?;
    }
    Ok(k)
}

pub(crate) fn check_mask(grid: &Grid, mask: Option<&[bool]>) -> Result<()> {
    match mask {
        Some(m) if m.len() != grid.n_cells() => invalid(format!(
            "mask has {} values for {} cells",
            m.len(),
            grid.n_cells()
        )),
        _ => Ok(()),
    }
}

pub(crate) fn check_quench(q: &QuenchOptions) -> Result<()> {
    if !(q.limit > 0.0 && q.limit <= 1.0) {
        return invalid(format!("quench limit must lie in (0, 1], got {}", q.limit));
    }
    if !(q.tol.is_finite() && q.tol >= 0.0) {
        return invalid(format!(
            "quench tol must be finite and non-negative, got {}",
            q.tol
        ));
    }
    if !(q.proportion_weight.is_finite() && q.proportion_weight >= 0.0) {
        return invalid(format!(
            "proportion_weight must be finite and non-negative, got {}",
            q.proportion_weight
        ));
    }
    Ok(())
}
