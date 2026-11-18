use varvite_core::{Mat, Result};

use crate::fit::MarkovChain;

/// Entropies in bits.
#[derive(Clone, Debug, PartialEq)]
pub struct Entropy {
    pub post: Vec<f64>,
    pub pre: Vec<f64>,
    pub post_normalized: Vec<f64>,
    pub pre_normalized: Vec<f64>,
    pub chain: f64,
    pub max: f64,
}

impl MarkovChain {
    pub fn entropy(&self) -> Result<Entropy> {
        todo!()
    }

    /// `F - F^T` with `F_jk = p_j q_jk` or `nu_j pi_jk`.
    pub fn asymmetry(&self) -> Result<Mat> {
        todo!()
    }
}
