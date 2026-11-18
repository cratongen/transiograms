//! Markov chain random field simulation and Markov-type categorical prediction (Li and Zhang
//! Eq. 14) on grids.

pub mod conditional;
pub mod predict;
pub mod search;
pub mod simulate;

pub use conditional::{conditional, conditional_eq13};
pub use predict::predict;
pub use simulate::simulate;

use varvite_core::{Code, Grid, MISSING, Result, Snapped, invalid};
use varvite_transiogram::TransitionModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Neighbourhood {
    /// Quadrants in 2D, octants in 3D.
    Sectors,
    Nearest(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub neighbourhood: Neighbourhood,
    pub closeness_limit: f64,
    pub n_realizations: usize,
    pub seed: u64,
    pub memory: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            neighbourhood: Neighbourhood::Sectors,
            closeness_limit: 0.01,
            n_realizations: 1,
            seed: 0,
            memory: None,
        }
    }
}

/// Validates the inputs and returns the hard codes laid out per cell (MISSING elsewhere).
pub(crate) fn hard_codes(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
) -> Result<Vec<Code>> {
    let n = grid.n_cells();
    let k = model.k();
    if model.proportions.iter().any(|&p| p.is_nan() || p <= 0.0) {
        return invalid("every proportion must be positive");
    }
    if let Some(m) = mask
        && m.len() != n
    {
        return invalid(format!(
            "mask has {} values but the grid has {n} cells",
            m.len()
        ));
    }
    if hard.cells.len() != hard.codes.len() {
        return invalid("hard data need one code per cell");
    }
    let mut codes = vec![MISSING; n];
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        if cell >= n {
            return invalid(format!("hard cell {cell} is outside the grid of {n} cells"));
        }
        if code as usize >= k {
            return invalid(format!("hard code {code} at cell {cell} is not in 0..{k}"));
        }
        if mask.is_some_and(|m| !m[cell]) {
            return invalid(format!("hard cell {cell} is outside the mask"));
        }
        codes[cell] = code;
    }
    Ok(codes)
}
