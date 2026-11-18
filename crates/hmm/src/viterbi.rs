use varvite_core::Result;

use crate::model::{Decoder, Hmm};
use crate::sequence::{Clamp, Sequence};

impl Hmm {
    pub fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<(Vec<u32>, f64)> {
        todo!()
    }

    pub fn decode(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        decoder: Decoder,
    ) -> Result<Vec<u32>> {
        todo!()
    }
}
