use serde::{Deserialize, Serialize};
use varvite_core::{Mat, RateMatrix, Result};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionModel {
    pub proportions: Vec<f64>,
    /// Principal x', y', z' (z' up), each the +direction generator.
    pub rates: [RateMatrix; 3],
    /// Azimuth, dip, rake in degrees.
    pub rotation: [f64; 3],
    pub background: usize,
}

impl TransitionModel {
    /// Each rate matrix valid with `p^T R = 0` within `1e-10 max|r|`.
    pub fn new(
        proportions: Vec<f64>,
        rates: [RateMatrix; 3],
        rotation: [f64; 3],
        background: usize,
    ) -> Result<Self> {
        todo!()
    }

    /// Per axis: embedded given -> background fill; None -> maximum entropy (symmetric).
    pub fn from_geology(
        proportions: &[f64],
        mean_lengths: [&[f64]; 3],
        embedded: [Option<&Mat>; 3],
        background: usize,
        rotation: [f64; 3],
    ) -> Result<Self> {
        todo!()
    }

    pub fn k(&self) -> usize {
        todo!()
    }

    pub fn to_principal(&self, h: [f64; 3]) -> [f64; 3] {
        todo!()
    }

    /// `h != 0`.
    pub fn rates_along(&self, h: [f64; 3]) -> Result<RateMatrix> {
        todo!()
    }

    /// `T(0) = I`.
    pub fn transition(&self, h: [f64; 3]) -> Mat {
        todo!()
    }

    /// `det T(h)^(1/(k-1))`; 1 when k = 1.
    pub fn closeness(&self, h: [f64; 3]) -> f64 {
        todo!()
    }

    pub fn mean_lengths(&self, h: [f64; 3]) -> Vec<f64> {
        todo!()
    }
}
