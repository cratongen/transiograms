use serde::{Deserialize, Serialize};
use varvite_core::{Code, Direction, Grid, Mat, Result, SequenceSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experimental {
    /// Nominal lag `l * Delta` (or the given lags).
    pub lags: Vec<f64>,
    /// Mean `|h|` of the pairs (NaN when none).
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

pub fn experimental(
    coords: &[[f64; 3]],
    codes: &[Code],
    holes: Option<&[u32]>,
    k: usize,
    spec: &PairSpec,
) -> Result<Experimental> {
    todo!()
}

pub fn downhole(
    seqs: &SequenceSet,
    direction: Direction,
    lags: &[f64],
    gap: f64,
) -> Result<Experimental> {
    todo!()
}

pub fn on_grid(
    codes: &[Code],
    grid: &Grid,
    mask: Option<&[bool]>,
    offset: [i64; 3],
    n_lags: usize,
    k: usize,
) -> Result<Experimental> {
    todo!()
}
