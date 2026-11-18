use varvite_core::Result;
use varvite_hmm::{Clamp, Family, FitOptions, Hmm, Sequence};

use crate::duration::DurationFamily;
use crate::model::Hsmm;

#[derive(Clone, Debug, PartialEq)]
pub enum Model {
    Hmm(Hmm),
    Hsmm(Hsmm),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Hmm,
    Hsmm {
        duration: DurationFamily,
        max_duration: Option<usize>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionSpec {
    pub states: Vec<usize>,
    pub family: Family,
    pub kind: ModelKind,
    pub fit: FitOptions,
    /// None = leave one hole out, Some(0) = no CV.
    pub cv_folds: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionTable {
    pub states: Vec<usize>,
    pub log_likelihood: Vec<f64>,
    pub n_params: Vec<usize>,
    pub n_observations: usize,
    pub aic: Vec<f64>,
    pub bic: Vec<f64>,
    pub icl: Vec<f64>,
    pub cv_log_likelihood: Vec<f64>,
    pub cv_per_step: Vec<f64>,
}

pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
    todo!()
}

pub fn bic(log_likelihood: f64, n_params: usize, n_observations: usize) -> f64 {
    todo!()
}

pub fn icl(log_likelihood: f64, n_params: usize, n_observations: usize, entropy: f64) -> f64 {
    todo!()
}

pub fn select_states(
    data: &[Sequence],
    clamps: Option<&[Clamp]>,
    spec: &SelectionSpec,
) -> Result<SelectionTable> {
    todo!()
}
