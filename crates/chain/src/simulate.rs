use varvite_core::{Code, Result};

use crate::fit::MarkovChain;

/// Hard intervals as positions along the direction.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Hard {
    pub from: Vec<f64>,
    pub to: Vec<f64>,
    pub codes: Vec<Code>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    Stationary,
    Code(Code),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Steps(usize),
    Beds(usize),
    Length(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Simulated {
    Codes(Vec<Vec<Code>>),
    Beds(Vec<(Vec<f64>, Vec<f64>, Vec<Code>)>),
}

impl MarkovChain {
    /// Realization `r` draws from `Stream::new(seed, &[r])`.
    pub fn simulate(
        &self,
        target: Target,
        hard: &Hard,
        start: Start,
        n: usize,
        seed: u64,
    ) -> Result<Simulated> {
        todo!()
    }
}
