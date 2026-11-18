use varvite_core::{Code, Direction, Result, SequenceSet};

use crate::tests::TestResult;

#[derive(Clone, Debug, PartialEq)]
pub struct Sojourn {
    pub n_beds: Vec<usize>,
    pub n_open: Vec<usize>,
    pub total: Vec<f64>,
    pub mean_length: Vec<f64>,
    pub thickness: Vec<Vec<f64>>,
    pub open: Vec<Vec<bool>>,
}

pub fn sojourn(seqs: &SequenceSet, direction: Direction, gap: f64) -> Sojourn {
    todo!()
}

impl Sojourn {
    /// Kaplan-Meier, open beds censored.
    pub fn survival(&self, code: Code) -> (Vec<f64>, Vec<f64>) {
        todo!()
    }

    pub fn exponential_test(
        &self,
        code: Code,
        n_bootstrap: usize,
        seed: u64,
    ) -> Result<TestResult> {
        todo!()
    }
}
