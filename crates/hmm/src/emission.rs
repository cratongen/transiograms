use serde::{Deserialize, Serialize};
use varvite_core::{Mat, Result, Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CovarianceType {
    Full,
    Diag,
    Spherical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Categorical {
    pub probabilities: Vec<f64>,
    pub pseudocount: f64,
    pub frozen: bool,
}

/// `covariance` is always `d x d`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gaussian {
    pub mean: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat")]
    pub covariance: Mat,
    pub covariance_type: CovarianceType,
    pub frozen: bool,
}

/// Independent channels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Poisson {
    pub rate: Vec<f64>,
    pub frozen: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianMixture {
    pub weights: Vec<f64>,
    pub components: Vec<Gaussian>,
    pub frozen: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "lowercase")]
pub enum Emission {
    Categorical(Categorical),
    Gaussian(Gaussian),
    Poisson(Poisson),
    Mixture(GaussianMixture),
}

impl Emission {
    /// 1 for categorical.
    pub fn dim(&self) -> usize {
        todo!()
    }

    /// Missing channels marginalised; all missing -> 0.
    pub fn log_density(&self, x: &[f64]) -> f64 {
        todo!()
    }

    /// 0 when frozen.
    pub fn n_params(&self) -> usize {
        todo!()
    }

    pub fn sample(&self, rng: &mut Stream) -> Vec<f64> {
        todo!()
    }

    pub fn validate(&self) -> Result<()> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Categorical {
        n_symbols: usize,
    },
    Gaussian {
        covariance_type: CovarianceType,
    },
    Poisson,
    Mixture {
        n_mix: usize,
        covariance_type: CovarianceType,
    },
}
