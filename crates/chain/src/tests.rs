//! Statistical tests on transition counts.

use serde::{Deserialize, Serialize};
use varvite_core::{Mat, Result, SequenceSet};

use crate::fit::{FitOptions, MarkovChain};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    G,
    Pearson,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    /// "independence", "order 2 vs 1", "symmetry", "homogeneity", "exponential".
    pub name: String,
    /// "g", "pearson", "bowker", "ks".
    pub method: String,
    pub statistic: f64,
    /// 0 for "ks".
    pub df: usize,
    pub p_value: f64,
    /// Null-hypothesis expected counts when tabular.
    #[serde(with = "varvite_core::serde_mat::option")]
    pub expected: Option<Mat>,
    /// Fraction of allowed cells with expected < 5.
    pub small_expected: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderTable {
    pub orders: Vec<usize>,
    pub log_likelihood: Vec<f64>,
    pub n_params: Vec<usize>,
    pub aic: Vec<f64>,
    pub bic: Vec<f64>,
    pub n_transitions: usize,
}

impl MarkovChain {
    pub fn independence_test(&self, method: Method) -> Result<TestResult> {
        todo!()
    }

    /// Bowker on counts.
    pub fn symmetry_test(&self) -> Result<TestResult> {
        todo!()
    }

    /// `Pi_obs - Pi_expected` under quasi-independence.
    pub fn difference_matrix(&self) -> Result<Mat> {
        todo!()
    }
}

pub fn order_test(
    seqs: &SequenceSet,
    opts: &FitOptions,
    order: usize,
    method: Method,
) -> Result<TestResult> {
    todo!()
}

/// Fixed only.
pub fn select_order(seqs: &SequenceSet, opts: &FitOptions, max_order: usize) -> Result<OrderTable> {
    todo!()
}

pub fn homogeneity_test(
    groups: &[SequenceSet],
    opts: &FitOptions,
    method: Method,
) -> Result<TestResult> {
    todo!()
}
