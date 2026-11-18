use varvite_core::{Code, Grid, Result};
use varvite_transiogram::TransitionModel;

use crate::QuenchOptions;

/// Sequential, zero temperature; returns the objective after each sweep (index 0 = initial).
#[allow(clippy::too_many_arguments)]
pub fn quench(
    codes: &mut [Code],
    frozen: &[bool],
    grid: &Grid,
    mask: Option<&[bool]>,
    model: &TransitionModel,
    q: &QuenchOptions,
    seed: u64,
    realization: u64,
) -> Result<Vec<f64>> {
    todo!()
}

pub fn objective(
    codes: &[Code],
    grid: &Grid,
    mask: Option<&[bool]>,
    model: &TransitionModel,
    lags: &[[i64; 3]],
    proportion_weight: f64,
) -> f64 {
    todo!()
}

pub fn quench_lags(model: &TransitionModel, grid: &Grid, limit: f64) -> Vec<[i64; 3]> {
    todo!()
}
