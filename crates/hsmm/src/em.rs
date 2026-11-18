use varvite_core::Result;
use varvite_hmm::{Clamp, Family, FitOptions, FitReport, Sequence};

use crate::duration::DurationFamily;
use crate::model::Hsmm;

impl Hsmm {
    pub fn fit(
        &mut self,
        family: &Family,
        duration: DurationFamily,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
    ) -> Result<FitReport> {
        todo!()
    }
}
