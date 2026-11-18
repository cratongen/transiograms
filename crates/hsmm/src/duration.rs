use serde::{Deserialize, Serialize};
use varvite_core::Result;

/// Shifted laws on u = 1, 2, ...; truncated and renormalised on `1..=max_duration`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum Duration {
    /// `p (1 - p)^(u - 1)`, `0 < p <= 1`.
    Geometric { p: f64 },
    /// `e^-rate rate^(u-1) / (u-1)!`, `rate >= 0`.
    Poisson { rate: f64 },
    /// `C(u + r - 2, u - 1) p^r (1 - p)^(u - 1)`.
    NegativeBinomial { r: f64, p: f64 },
    /// `pmf[u - 1]`; support `1..=pmf.len()`.
    Empirical { pmf: Vec<f64> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurationDist {
    pub law: Duration,
    pub frozen: bool,
    pub pseudocount: f64,
}

impl Duration {
    /// Index `u - 1`, renormalised on `1..=m`.
    pub fn log_pmf(&self, m: usize) -> Vec<f64> {
        todo!()
    }

    /// `log D(u) = log sum_{v >= u} d(v)`.
    pub fn log_survivor(&self, m: usize) -> Vec<f64> {
        todo!()
    }

    pub fn mean(&self, m: usize) -> f64 {
        todo!()
    }

    /// 1, 1, 2, m - 1.
    pub fn n_params(&self, m: usize) -> usize {
        todo!()
    }

    /// Weighted MLE from `counts[u - 1]`.
    pub fn fit(&self, counts: &[f64], pseudocount: f64) -> Result<Duration> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationFamily {
    Geometric,
    Poisson,
    NegativeBinomial,
    Empirical,
}
