use varvite_core::{Mat, Result};
use varvite_hmm::{Clamp, Sequence};

use crate::model::Hsmm;

#[derive(Clone, Debug, PartialEq)]
pub struct SegmentPosterior {
    pub log_likelihood: f64,
    pub gamma: Mat,
    pub xi_sum: Mat,
    pub start: Vec<f64>,
    pub durations: Vec<Vec<f64>>,
    pub entropy: f64,
}

impl Hsmm {
    pub fn log_likelihood(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<f64> {
        todo!()
    }

    pub fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<SegmentPosterior> {
        todo!()
    }

    pub fn sample_posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        n: usize,
        seed: u64,
        index: u64,
    ) -> Result<Vec<Vec<u32>>> {
        todo!()
    }
}
