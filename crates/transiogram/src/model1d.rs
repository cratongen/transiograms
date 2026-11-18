use serde::{Deserialize, Serialize};
use varvite_core::{Mat, RateMatrix, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Structure {
    Exponential,
    Spherical,
    Gaussian,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub structure: Structure,
    pub range: f64,
    pub nugget: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Model1D {
    Markov {
        proportions: Vec<f64>,
        rates: RateMatrix,
    },
    /// `entries`: k * k, None at the complement entry of each row.
    Entrywise {
        proportions: Vec<f64>,
        entries: Vec<Option<Entry>>,
        complement: Vec<usize>,
    },
}

impl Model1D {
    /// Proportions = stationary law of `rates`.
    pub fn markov(rates: RateMatrix) -> Result<Self> {
        todo!()
    }

    /// Validated on a lag grid up to `max_lag`.
    pub fn entrywise(
        proportions: Vec<f64>,
        entries: Vec<Option<Entry>>,
        complement: Vec<usize>,
        max_lag: f64,
    ) -> Result<Self> {
        todo!()
    }

    /// `h < 0` by the reversal identity.
    pub fn transition(&self, h: f64) -> Mat {
        todo!()
    }

    pub fn proportions(&self) -> &[f64] {
        todo!()
    }
}
