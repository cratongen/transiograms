use serde::{Deserialize, Serialize};
use transiograms_core::{Mat, Result};

use crate::fit::MarkovChain;

/// Entropies in bits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entropy {
    #[serde(with = "transiograms_core::serde_mat::values")]
    pub post: Vec<f64>,
    #[serde(with = "transiograms_core::serde_mat::values")]
    pub pre: Vec<f64>,
    #[serde(with = "transiograms_core::serde_mat::values")]
    pub post_normalized: Vec<f64>,
    #[serde(with = "transiograms_core::serde_mat::values")]
    pub pre_normalized: Vec<f64>,
    pub chain: f64,
    pub max: f64,
}

/// Shannon entropy in bits; a certain outcome gives +0.0, not -0.0.
fn h(ps: impl Iterator<Item = f64>) -> f64 {
    0.0 - ps.filter(|&p| p > 0.0).map(|p| p * p.log2()).sum::<f64>()
}

impl MarkovChain {
    /// Hattori (1976). `pre_j` is NaN for a category never entered.
    pub fn entropy(&self) -> Result<Entropy> {
        let m = self.k();
        let p = self.transition.matrix();
        let nu = self.transition.stationary()?;
        let post: Vec<f64> = (0..m).map(|i| h(p.row(i).iter().copied())).collect();
        let r = match &self.counts {
            Some(n) => n.clone(),
            None => Mat::from_fn(m, m, |i, j| nu[i] * p[(i, j)]),
        };
        let pre: Vec<f64> = (0..m)
            .map(|j| {
                let s: f64 = r.column(j).sum();
                if s > 0.0 {
                    h(r.column(j).iter().map(|&v| v / s))
                } else {
                    f64::NAN
                }
            })
            .collect();
        let max = if self.transition.is_embedded() {
            ((m - 1) as f64).log2()
        } else {
            (m as f64).log2()
        };
        let norm = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .map(|&x| if max > 0.0 { x / max } else { 0.0 })
                .collect()
        };
        Ok(Entropy {
            post_normalized: norm(&post),
            pre_normalized: norm(&pre),
            chain: (0..m).map(|i| nu[i] * post[i]).sum(),
            post,
            pre,
            max,
        })
    }

    /// `F - F^T` with `F_jk = p_j q_jk` or `nu_j pi_jk`.
    pub fn asymmetry(&self) -> Result<Mat> {
        let f = match &self.generator {
            Some(q) => {
                let p = q.stationary()?;
                Mat::from_fn(self.k(), self.k(), |j, k| p[j] * q.matrix()[(j, k)])
            }
            None => {
                let nu = self.transition.stationary()?;
                let pi = self.transition.matrix();
                Mat::from_fn(self.k(), self.k(), |j, k| nu[j] * pi[(j, k)])
            }
        };
        Ok(&f - f.transpose())
    }
}
