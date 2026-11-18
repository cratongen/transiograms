use varvite_core::{RateMatrix, Result};

use crate::experimental::Experimental;
use crate::model1d::{Model1D, Structure};

impl Model1D {
    pub fn fit_markov(
        exp: &Experimental,
        proportions: Option<&[f64]>,
        background: usize,
        start: Option<&RateMatrix>,
    ) -> Result<Self> {
        todo!()
    }

    pub fn fit_entrywise(
        exp: &Experimental,
        proportions: &[f64],
        structures: &[Structure],
        complement: &[usize],
    ) -> Result<Self> {
        todo!()
    }
}
