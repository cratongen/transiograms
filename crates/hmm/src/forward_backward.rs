use varvite_core::{Mat, Result};

use crate::model::{Hmm, Implementation};
use crate::sequence::{Clamp, Sequence};

#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    pub log_likelihood: f64,
    /// `t x K`.
    pub gamma: Mat,
    /// `K x K`, sum over t of `xi_t`.
    pub xi_sum: Mat,
    /// `H(S | X)` of the whole path.
    pub entropy: f64,
}

impl Hmm {
    pub fn log_likelihood(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        imp: Implementation,
    ) -> Result<f64> {
        todo!()
    }

    pub fn posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        imp: Implementation,
    ) -> Result<Posterior> {
        todo!()
    }

    /// Stream `(seed, [SAMPLE, index])`, `n` draws sequential.
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
