use varvite_core::Result;

use crate::emission::Family;
use crate::model::{FitOptions, FitReport, Hmm};
use crate::sequence::{Clamp, Sequence};

impl Hmm {
    /// Restarts are fit in parallel; restart `r` uses `Stream::new(seed, &[tag::INIT, r])`.
    pub fn fit(
        &mut self,
        family: &Family,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
    ) -> Result<FitReport> {
        todo!()
    }
}
