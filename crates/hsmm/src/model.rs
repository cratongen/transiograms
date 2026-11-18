use serde::{Deserialize, Serialize};
use varvite_core::{Mat, Result, Stream};
use varvite_hmm::{Clamp, Emission, Family, Hmm, Params, Priors, Sequence};

use crate::duration::{DurationDist, DurationFamily};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hsmm {
    pub start: Vec<f64>,
    /// Zero diagonal; K = 1 allowed (`[[0]]`).
    #[serde(with = "varvite_core::serde_mat")]
    pub transition: Mat,
    pub durations: Vec<DurationDist>,
    pub emissions: Vec<Emission>,
    pub max_duration: Vec<usize>,
}

impl Hsmm {
    pub fn new(
        start: Vec<f64>,
        transition: Mat,
        durations: Vec<DurationDist>,
        emissions: Vec<Emission>,
        max_duration: Vec<usize>,
    ) -> Result<Self> {
        todo!()
    }

    /// Geometric equivalence: `d_j` geometric with `p = 1 - a_jj`, `p_ij = a_ij / (1 - a_jj)`.
    pub fn from_hmm(hmm: &Hmm, max_duration: usize) -> Result<Self> {
        todo!()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        k: usize,
        family: &Family,
        duration: DurationFamily,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        priors: &Priors,
        max_duration: &[usize],
        stream: &mut Stream,
    ) -> Result<Self> {
        todo!()
    }

    pub fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)> {
        todo!()
    }

    pub fn n_params(&self, params: Params) -> usize {
        todo!()
    }

    /// Untruncated law's mass above `max_duration`.
    pub fn truncated_mass(&self) -> Vec<f64> {
        todo!()
    }
}
