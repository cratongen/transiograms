use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use transiograms_core::rng::tag;
use transiograms_core::{Mat, Result, Stream, invalid, memory};

/// Upper bound on the `t x K` f64 matrices one forward-backward or Viterbi pass holds at once.
const PASS_MATRICES: usize = 6;

use crate::emission::{CovarianceType, Emission, Gaussian, check_simplex, chol_logdet};
use crate::sequence::{Clamp, Sequence, check_sequence};

/// Letters: s start, t transition, e emissions, d durations (HSMM only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub start: bool,
    pub transition: bool,
    pub emission: bool,
    pub duration: bool,
}

impl Params {
    pub fn parse(letters: &str) -> Result<Self> {
        let mut p = Params {
            start: false,
            transition: false,
            emission: false,
            duration: false,
        };
        for c in letters.chars() {
            match c {
                's' => p.start = true,
                't' => p.transition = true,
                'e' => p.emission = true,
                'd' => p.duration = true,
                _ => {
                    return invalid(format!(
                        "unknown parameter letter {c:?} in {letters:?}; use s, t, e and d"
                    ));
                }
            }
        }
        Ok(p)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Implementation {
    Log,
    Scaled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    None,
    Mean,
    Occupancy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    Viterbi,
    Map,
    PosteriorViterbi,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Priors {
    pub start: f64,
    pub transition: f64,
    pub covariance: f64,
}

impl Default for Priors {
    fn default() -> Self {
        Self {
            start: 1.0,
            transition: 1.0,
            covariance: 1e-6,
        }
    }
}

impl Priors {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(self.start >= 1.0 && self.start.is_finite())
            || !(self.transition >= 1.0 && self.transition.is_finite())
        {
            return invalid(format!(
                "start and transition priors must be finite and at least 1, not {} and {}",
                self.start, self.transition
            ));
        }
        if !(self.covariance >= 0.0 && self.covariance.is_finite()) {
            return invalid(format!(
                "covariance prior must be finite and non-negative, not {}",
                self.covariance
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    pub max_iter: usize,
    /// Objective gain per observed step.
    pub tol: f64,
    pub n_init: usize,
    pub seed: u64,
    pub params: Params,
    pub init_params: Params,
    pub priors: Priors,
    pub implementation: Implementation,
    pub ordering: Ordering,
}

impl Default for FitOptions {
    fn default() -> Self {
        let ste = Params {
            start: true,
            transition: true,
            emission: true,
            duration: false,
        };
        Self {
            max_iter: 100,
            tol: 1e-6,
            n_init: 1,
            seed: 0,
            params: ste,
            init_params: ste,
            priors: Priors::default(),
            implementation: Implementation::Log,
            ordering: Ordering::None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitReport {
    pub history: Vec<f64>,
    pub converged: bool,
    pub n_iter: usize,
    pub restart: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hmm {
    pub start: Vec<f64>,
    #[serde(with = "transiograms_core::serde_mat")]
    pub transition: Mat,
    pub emissions: Vec<Emission>,
}

impl Hmm {
    pub fn new(start: Vec<f64>, transition: Mat, emissions: Vec<Emission>) -> Result<Self> {
        let hmm = Self {
            start,
            transition,
            emissions,
        };
        hmm.validate()?;
        Ok(hmm)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let k = self.k();
        if k == 0 {
            return invalid("an HMM needs at least one state");
        }
        check_simplex(&self.start, "start probabilities")?;
        if self.transition.shape() != (k, k) {
            return invalid(format!(
                "transition must be {k} x {k}, not {} x {}",
                self.transition.nrows(),
                self.transition.ncols()
            ));
        }
        for i in 0..k {
            let row: Vec<f64> = self.transition.row(i).iter().copied().collect();
            check_simplex(&row, &format!("transition row {i}"))?;
        }
        validate_emissions(&self.emissions, k)
    }

    pub fn k(&self) -> usize {
        self.start.len()
    }

    /// Validates the model too: public fields and deserialised JSON bypass `Hmm::new`.
    pub(crate) fn check(&self, seq: &Sequence, clamp: Option<&Clamp>, n: usize) -> Result<()> {
        self.validate()?;
        self.check_sequence(seq, clamp, n)
    }

    pub(crate) fn check_sequence(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        n: usize,
    ) -> Result<()> {
        let e = &self.emissions[0];
        check_sequence(seq, clamp, n, e.dim(), self.k(), e.symbols())?;
        let bytes = seq
            .t
            .checked_mul(self.k())
            .and_then(|tk| tk.checked_mul(PASS_MATRICES * size_of::<f64>()));
        match bytes {
            Some(b) => memory::check(b as u64, None, &format!("sequence {n} pass")),
            None => invalid(format!("sequence {n}: t x K overflows usize")),
        }
    }

    /// Stream `(seed, [SAMPLE, i])` for sequence `i`.
    pub fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)> {
        let d = self.emissions[0].dim();
        lengths
            .par_iter()
            .enumerate()
            .map(|(i, &n)| {
                let mut stream = Stream::new(seed, &[tag::SAMPLE, i as u64]);
                let mut states = Vec::with_capacity(n);
                let mut x = Vec::with_capacity(n * d);
                let mut s = 0;
                for t in 0..n {
                    s = if t == 0 {
                        stream.categorical(&self.start)
                    } else {
                        let row: Vec<f64> = self.transition.row(s).iter().copied().collect();
                        stream.categorical(&row)
                    };
                    states.push(s as u32);
                    x.extend(self.emissions[s].sample(&mut stream));
                }
                (states, Sequence { t: n, d, x })
            })
            .collect()
    }

    pub fn n_params(&self, params: Params) -> usize {
        let mut n = 0;
        if params.start {
            n += self
                .start
                .iter()
                .filter(|&&p| p > 0.0)
                .count()
                .saturating_sub(1);
        }
        if params.transition {
            for i in 0..self.k() {
                n += self
                    .transition
                    .row(i)
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .count()
                    .saturating_sub(1);
            }
        }
        if params.emission {
            n += self.emissions.iter().map(Emission::n_params).sum::<usize>();
        }
        n
    }

    pub fn log_prior(&self, priors: &Priors, pooled_variance: &[f64]) -> f64 {
        let mut lp = 0.0;
        if priors.start != 1.0 {
            lp += (priors.start - 1.0) * sum_ln_positive(self.start.iter());
        }
        if priors.transition != 1.0 {
            lp += (priors.transition - 1.0) * sum_ln_positive(self.transition.iter());
        }
        for e in &self.emissions {
            lp += emission_log_prior(e, priors.covariance, pooled_variance);
        }
        lp
    }
}

pub(crate) fn validate_emissions(emissions: &[Emission], k: usize) -> Result<()> {
    if emissions.len() != k {
        return invalid(format!(
            "give {k} emission distributions, not {}",
            emissions.len()
        ));
    }
    let first = &emissions[0];
    for e in emissions {
        e.validate()?;
        if std::mem::discriminant(e) != std::mem::discriminant(first)
            || e.dim() != first.dim()
            || e.symbols() != first.symbols()
        {
            return invalid(
                "every state needs the same emission family, dimension and symbol count",
            );
        }
    }
    Ok(())
}

/// The emission part of `Hmm::log_prior` for one state, with covariance prior weight `eps`;
/// shared with the HSMM.
pub fn emission_log_prior(e: &Emission, eps: f64, pooled_variance: &[f64]) -> f64 {
    match e {
        Emission::Categorical(c) if c.pseudocount > 0.0 => {
            c.pseudocount * sum_ln_positive(c.probabilities.iter())
        }
        Emission::Gaussian(g) if eps > 0.0 => cov_kernel(g, eps, pooled_variance),
        Emission::Mixture(m) if eps > 0.0 => m
            .components
            .iter()
            .map(|g| cov_kernel(g, eps, pooled_variance))
            .sum(),
        _ => 0.0,
    }
}

fn sum_ln_positive<'a>(v: impl Iterator<Item = &'a f64>) -> f64 {
    v.filter(|&&p| p > 0.0).map(|p| p.ln()).sum()
}

/// `-(1/2) ln|S| - (1/2) tr(S^-1 Psi)` with `Psi = eps diag(v)`.
fn cov_kernel(g: &Gaussian, eps: f64, v: &[f64]) -> f64 {
    let d = g.mean.len();
    let s = &g.covariance;
    let vc = |c: usize| eps * v.get(c).copied().unwrap_or(0.0);
    match g.covariance_type {
        CovarianceType::Full => {
            let Some((l, logdet)) = chol_logdet(s) else {
                return f64::NEG_INFINITY;
            };
            let mut tr = 0.0;
            for c in 0..d {
                let mut e = vec![0.0; d];
                e[c] = 1.0;
                crate::emission::forward_solve(&l, &mut e);
                tr += vc(c) * e.iter().map(|x| x * x).sum::<f64>();
            }
            -0.5 * logdet - 0.5 * tr
        }
        CovarianceType::Diag => (0..d)
            .map(|c| -0.5 * s[(c, c)].ln() - 0.5 * vc(c) / s[(c, c)])
            .sum(),
        CovarianceType::Spherical => {
            let s2 = s[(0, 0)];
            -0.5 * d as f64 * s2.ln() - 0.5 * (0..d).map(vc).sum::<f64>() / s2
        }
    }
}

/// Steps with at least one observed channel, over all sequences.
pub fn n_observed(data: &[Sequence]) -> usize {
    data.iter()
        .map(|s| {
            (0..s.t)
                .filter(|&t| s.row(t).iter().any(|v| !v.is_nan()))
                .count()
        })
        .sum()
}
