use serde::{Deserialize, Serialize};
use varvite_core::{Mat, Result};

use crate::emission::Emission;
use crate::sequence::Sequence;

/// Letters: s start, t transition, e emissions, d durations (HSMM only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub start: bool,
    pub transition: bool,
    pub emission: bool,
    pub duration: bool,
}

impl Params {
    pub fn parse(letters: &str) -> Result<Self> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Implementation {
    Log,
    Scaled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    None,
    Mean,
    Occupancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    Viterbi,
    Map,
    PosteriorViterbi,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Priors {
    pub start: f64,
    pub transition: f64,
    pub covariance: f64,
}

impl Default for Priors {
    fn default() -> Self {
        Self {
            start: 1.0,
            transition: 1.0,
            covariance: 1e-6,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    pub max_iter: usize,
    /// Objective gain per observed step.
    pub tol: f64,
    pub n_init: usize,
    pub seed: u64,
    pub params: Params,
    pub init_params: Params,
    pub priors: Priors,
    pub implementation: Implementation,
    pub ordering: Ordering,
}

impl Default for FitOptions {
    fn default() -> Self {
        let ste = Params {
            start: true,
            transition: true,
            emission: true,
            duration: false,
        };
        Self {
            max_iter: 100,
            tol: 1e-6,
            n_init: 1,
            seed: 0,
            params: ste,
            init_params: ste,
            priors: Priors::default(),
            implementation: Implementation::Log,
            ordering: Ordering::None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitReport {
    pub history: Vec<f64>,
    pub converged: bool,
    pub n_iter: usize,
    pub restart: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hmm {
    pub start: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat")]
    pub transition: Mat,
    pub emissions: Vec<Emission>,
}

impl Hmm {
    pub fn new(start: Vec<f64>, transition: Mat, emissions: Vec<Emission>) -> Result<Self> {
        todo!()
    }

    pub fn k(&self) -> usize {
        todo!()
    }

    /// Stream `(seed, [SAMPLE, i])` for sequence `i`.
    pub fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)> {
        todo!()
    }

    pub fn n_params(&self, params: Params) -> usize {
        todo!()
    }

    pub fn log_prior(&self, priors: &Priors, pooled_variance: &[f64]) -> f64 {
        todo!()
    }
}

/// Steps with at least one observed channel, over all sequences.
pub fn n_observed(data: &[Sequence]) -> usize {
    todo!()
}
