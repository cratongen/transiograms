use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use varvite_core::nalgebra::Cholesky;
use varvite_core::numerics::logsumexp;
use varvite_core::special::ln_gamma;
use varvite_core::{Mat, ROW_TOL, Result, Stream, invalid};

const LN_2PI: f64 = 1.837_877_066_409_345_5;

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
        match self {
            Emission::Categorical(_) => 1,
            Emission::Gaussian(g) => g.mean.len(),
            Emission::Poisson(p) => p.rate.len(),
            Emission::Mixture(m) => m.components.first().map_or(0, |g| g.mean.len()),
        }
    }

    /// Missing channels marginalised; all missing -> 0.
    pub fn log_density(&self, x: &[f64]) -> f64 {
        Density::new(self).log_density(x)
    }

    /// 0 when frozen.
    pub fn n_params(&self) -> usize {
        let d = self.dim();
        match self {
            Emission::Categorical(c) if !c.frozen => c.probabilities.len().saturating_sub(1),
            Emission::Gaussian(g) if !g.frozen => d + cov_params(g.covariance_type, d),
            Emission::Poisson(p) if !p.frozen => d,
            Emission::Mixture(m) if !m.frozen => {
                let n = m.components.len();
                let ct = m
                    .components
                    .first()
                    .map_or(CovarianceType::Full, |g| g.covariance_type);
                n.saturating_sub(1) + n * d + n * cov_params(ct, d)
            }
            _ => 0,
        }
    }

    pub fn sample(&self, rng: &mut Stream) -> Vec<f64> {
        match self {
            Emission::Categorical(c) => vec![rng.categorical(&c.probabilities) as f64],
            Emission::Gaussian(g) => sample_gaussian(g, rng),
            Emission::Poisson(p) => p.rate.iter().map(|&l| sample_poisson(l, rng)).collect(),
            Emission::Mixture(m) => {
                sample_gaussian(&m.components[rng.categorical(&m.weights)], rng)
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Emission::Categorical(c) => {
                check_simplex(&c.probabilities, "categorical probabilities")?;
                if !(c.pseudocount.is_finite() && c.pseudocount >= 0.0) {
                    return invalid(format!(
                        "pseudocount must be finite and non-negative, not {}",
                        c.pseudocount
                    ));
                }
                Ok(())
            }
            Emission::Gaussian(g) => g.validate(),
            Emission::Poisson(p) => {
                if p.rate.is_empty() {
                    return invalid("a Poisson emission needs at least one rate");
                }
                if p.rate.iter().any(|r| !(r.is_finite() && *r >= 0.0)) {
                    return invalid(format!(
                        "Poisson rates must be finite and non-negative: {:?}",
                        p.rate
                    ));
                }
                Ok(())
            }
            Emission::Mixture(m) => {
                if m.components.is_empty() || m.weights.len() != m.components.len() {
                    return invalid(format!(
                        "a mixture needs one weight per component and at least one component, \
                         got {} weights and {} components",
                        m.weights.len(),
                        m.components.len()
                    ));
                }
                check_simplex(&m.weights, "mixture weights")?;
                let (d, ct) = (m.components[0].mean.len(), m.components[0].covariance_type);
                for g in &m.components {
                    g.validate()?;
                    if g.mean.len() != d || g.covariance_type != ct {
                        return invalid(
                            "mixture components must share dimension and covariance type",
                        );
                    }
                }
                Ok(())
            }
        }
    }

    pub fn frozen(&self) -> bool {
        match self {
            Emission::Categorical(c) => c.frozen,
            Emission::Gaussian(g) => g.frozen,
            Emission::Poisson(p) => p.frozen,
            Emission::Mixture(m) => m.frozen,
        }
    }

    pub(crate) fn symbols(&self) -> Option<usize> {
        match self {
            Emission::Categorical(c) => Some(c.probabilities.len()),
            _ => None,
        }
    }
}

impl Gaussian {
    fn validate(&self) -> Result<()> {
        let d = self.mean.len();
        if d == 0 {
            return invalid("a Gaussian emission needs at least one channel");
        }
        if self.mean.iter().any(|m| !m.is_finite()) {
            return invalid(format!("Gaussian mean must be finite: {:?}", self.mean));
        }
        let s = &self.covariance;
        if s.shape() != (d, d) {
            return invalid(format!(
                "covariance must be {d} x {d}, not {} x {}",
                s.nrows(),
                s.ncols()
            ));
        }
        if s.iter().any(|v| !v.is_finite()) {
            return invalid("covariance must be finite");
        }
        let scale = s.amax();
        for i in 0..d {
            for j in 0..i {
                if (s[(i, j)] - s[(j, i)]).abs() > 1e-12 * scale {
                    return invalid("covariance must be symmetric");
                }
                if self.covariance_type != CovarianceType::Full && s[(i, j)] != 0.0 {
                    return invalid("diag and spherical covariances must have zero off-diagonals");
                }
            }
            if self.covariance_type == CovarianceType::Spherical
                && (s[(i, i)] - s[(0, 0)]).abs() > 1e-12 * scale
            {
                return invalid("a spherical covariance must have equal diagonal entries");
            }
        }
        if Cholesky::new(s.clone()).is_none() {
            return invalid("covariance must be positive definite");
        }
        Ok(())
    }
}

pub(crate) fn cov_params(ct: CovarianceType, d: usize) -> usize {
    match ct {
        CovarianceType::Full => d * (d + 1) / 2,
        CovarianceType::Diag => d,
        CovarianceType::Spherical => 1,
    }
}

pub(crate) fn check_simplex(p: &[f64], what: &str) -> Result<()> {
    if p.is_empty() {
        return invalid(format!("{what} must not be empty"));
    }
    if p.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        return invalid(format!("{what} must be finite and non-negative: {p:?}"));
    }
    let sum: f64 = p.iter().sum();
    if (sum - 1.0).abs() > ROW_TOL * p.len().max(1) as f64 {
        return invalid(format!("{what} must sum to 1, not {sum}"));
    }
    Ok(())
}

/// Lower Cholesky factor and log-determinant.
pub(crate) fn chol_logdet(s: &Mat) -> Option<(Mat, f64)> {
    let l = Cholesky::new(s.clone())?.unpack();
    let logdet = 2.0 * l.diagonal().iter().map(|v| v.ln()).sum::<f64>();
    Some((l, logdet))
}

pub(crate) fn forward_solve(l: &Mat, r: &mut [f64]) {
    for i in 0..r.len() {
        let mut s = r[i];
        for j in 0..i {
            s -= l[(i, j)] * r[j];
        }
        r[i] = s / l[(i, i)];
    }
}

pub(crate) fn observed(x: &[f64]) -> Vec<usize> {
    (0..x.len()).filter(|&c| !x[c].is_nan()).collect()
}

/// Gaussian density with the Cholesky factor cached per missingness pattern.
pub(crate) struct GaussDensity<'a> {
    g: &'a Gaussian,
    full: Option<(Mat, f64)>,
    patterns: HashMap<Vec<usize>, Option<(Mat, f64)>>,
}

impl<'a> GaussDensity<'a> {
    pub(crate) fn new(g: &'a Gaussian) -> Self {
        Self {
            g,
            full: chol_logdet(&g.covariance),
            patterns: HashMap::new(),
        }
    }

    pub(crate) fn log_density(&mut self, x: &[f64]) -> f64 {
        let d = x.len();
        let obs = observed(x);
        if obs.is_empty() {
            return 0.0;
        }
        let factor = if obs.len() == d {
            self.full.as_ref()
        } else {
            let s = &self.g.covariance;
            self.patterns
                .entry(obs.clone())
                .or_insert_with(|| chol_logdet(&s.select_rows(&obs).select_columns(&obs)))
                .as_ref()
        };
        let Some((l, logdet)) = factor else {
            return f64::NEG_INFINITY;
        };
        let mut z: Vec<f64> = obs.iter().map(|&c| x[c] - self.g.mean[c]).collect();
        forward_solve(l, &mut z);
        let q: f64 = z.iter().map(|v| v * v).sum();
        -0.5 * (obs.len() as f64 * LN_2PI + logdet + q)
    }
}

pub(crate) enum Density<'a> {
    Categorical(Vec<f64>),
    Gaussian(GaussDensity<'a>),
    Poisson(&'a Poisson),
    Mixture(Vec<(f64, GaussDensity<'a>)>),
}

impl<'a> Density<'a> {
    pub(crate) fn new(e: &'a Emission) -> Self {
        match e {
            Emission::Categorical(c) => {
                Density::Categorical(c.probabilities.iter().map(|p| p.ln()).collect())
            }
            Emission::Gaussian(g) => Density::Gaussian(GaussDensity::new(g)),
            Emission::Poisson(p) => Density::Poisson(p),
            Emission::Mixture(m) => Density::Mixture(
                m.weights
                    .iter()
                    .zip(&m.components)
                    .map(|(w, g)| (w.ln(), GaussDensity::new(g)))
                    .collect(),
            ),
        }
    }

    pub(crate) fn log_density(&mut self, x: &[f64]) -> f64 {
        match self {
            Density::Categorical(lp) => {
                let v = x[0];
                if v.is_nan() {
                    0.0
                } else if v >= 0.0 && v.fract() == 0.0 && (v as usize) < lp.len() {
                    lp[v as usize]
                } else {
                    f64::NEG_INFINITY
                }
            }
            Density::Gaussian(g) => g.log_density(x),
            Density::Poisson(p) => {
                let mut s = 0.0;
                for (&v, &rate) in x.iter().zip(&p.rate) {
                    if v.is_nan() {
                        continue;
                    }
                    if v < 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    let lam = rate.max(1e-300);
                    s += v * lam.ln() - lam - ln_gamma(v + 1.0);
                }
                s
            }
            Density::Mixture(parts) => {
                if x.iter().all(|v| v.is_nan()) {
                    return 0.0;
                }
                let terms: Vec<f64> = parts
                    .iter_mut()
                    .map(|(lw, g)| *lw + g.log_density(x))
                    .collect();
                logsumexp(&terms)
            }
        }
    }
}

fn sample_gaussian(g: &Gaussian, rng: &mut Stream) -> Vec<f64> {
    let s = &g.covariance;
    let l = Cholesky::new(s.clone())
        .map(|c| c.unpack())
        .unwrap_or_else(|| Mat::from_diagonal(&s.diagonal().map(|v| v.max(0.0).sqrt())));
    let z: Vec<f64> = (0..g.mean.len()).map(|_| rng.normal()).collect();
    (0..g.mean.len())
        .map(|i| g.mean[i] + (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>())
        .collect()
}

fn sample_poisson(lam: f64, rng: &mut Stream) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    let u = rng.uniform();
    let ln_lam = lam.ln();
    let mut k = 0.0;
    let mut cdf = (-lam).exp();
    while cdf <= u {
        k += 1.0;
        let pmf = (k * ln_lam - lam - ln_gamma(k + 1.0)).exp();
        if pmf == 0.0 && k > lam {
            break;
        }
        cdf += pmf;
    }
    k
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
