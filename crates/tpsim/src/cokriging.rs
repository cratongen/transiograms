use varvite_core::{Code, Grid, Prediction, Result, Snapped};
use varvite_transiogram::TransitionModel;

use crate::Options;

/// Probabilities before order-relation correction, summing to 1.
pub fn cokrige(
    model: &TransitionModel,
    data: &[([f64; 3], Code)],
    target: [f64; 3],
    svd_cutoff: f64,
) -> Result<Vec<f64>> {
    todo!()
}

/// Clip to [0, 1] and renormalise; true if changed.
pub fn correct(probabilities: &mut [f64]) -> bool {
    todo!()
}

pub fn predict(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Prediction> {
    todo!()
}
