use varvite_core::Result;
use varvite_hmm::{Clamp, Sequence};

use crate::model::Hsmm;

impl Hsmm {
    pub fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<(Vec<u32>, f64)> {
        todo!()
    }
}
