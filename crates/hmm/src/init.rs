use varvite_core::{Mat, Result, Stream};

use crate::emission::Family;
use crate::model::{Hmm, Priors};
use crate::sequence::{Clamp, Sequence};

impl Hmm {
    /// k-means (continuous) or Dirichlet / supervised (categorical) start.
    pub fn initialize(
        k: usize,
        family: &Family,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        priors: &Priors,
        stream: &mut Stream,
    ) -> Result<Self> {
        todo!()
    }
}

pub fn kmeans(points: &Mat, k: usize, stream: &mut Stream) -> Vec<usize> {
    todo!()
}
