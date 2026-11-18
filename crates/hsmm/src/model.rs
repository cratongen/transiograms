use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use varvite_core::rng::tag;
use varvite_core::{Mat, Result, Stream, invalid, memory};
use varvite_hmm::{Clamp, Emission, Family, Hmm, Params, Priors, Sequence};

use crate::duration::{Duration, DurationDist, DurationFamily};
use crate::forward_backward::log_b;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hsmm {
    pub start: Vec<f64>,
    /// Zero diagonal; K = 1 allowed (`[[0]]`).
    #[serde(with = "varvite_core::serde_mat")]
    pub transition: Mat,
    pub durations: Vec<DurationDist>,
    pub emissions: Vec<Emission>,
    pub max_duration: Vec<usize>,
}

/// Log-space model tables shared by forward-backward, Viterbi and sampling.
#[derive(Clone, Debug)]
pub(crate) struct Tables {
    pub k: usize,
    pub log_pi: Vec<f64>,
    /// `-inf` on the diagonal.
    pub log_p: Mat,
    pub log_d: Vec<Vec<f64>>,
    pub log_dd: Vec<Vec<f64>>,
}

impl Tables {
    pub fn new(
        start: &[f64],
        transition: &Mat,
        laws: &[&Duration],
        max_duration: &[usize],
    ) -> Self {
        let k = start.len();
        let log_p = Mat::from_fn(k, k, |i, j| {
            if i == j {
                f64::NEG_INFINITY
            } else {
                transition[(i, j)].ln()
            }
        });
        Tables {
            k,
            log_pi: start.iter().map(|p| p.ln()).collect(),
            log_p,
            log_d: laws
                .iter()
                .zip(max_duration)
                .map(|(l, &m)| l.log_pmf(m))
                .collect(),
            log_dd: laws
                .iter()
                .zip(max_duration)
                .map(|(l, &m)| l.log_survivor(m))
                .collect(),
        }
    }

    pub fn m(&self, j: usize) -> usize {
        self.log_d[j].len()
    }
}

fn check_probabilities(v: &[f64], what: &str) -> Result<()> {
    if v.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return invalid(format!("{what} must be finite and non-negative"));
    }
    let s: f64 = v.iter().sum();
    if (s - 1.0).abs() > 1e-9 {
        return invalid(format!("{what} must sum to 1, not {s}"));
    }
    Ok(())
}

/// Per (state, u) the model holds `log_d`, `log_dd`, a raw pmf and duration counts.
const DURATION_TABLES: usize = 4;

/// Rejects `max_duration` of 0 and duration tables above the memory budget, before allocating.
pub(crate) fn check_max_duration(max_duration: &[usize]) -> Result<()> {
    if let Some(j) = max_duration.iter().position(|&m| m == 0) {
        return invalid(format!("max_duration must be at least 1 (state {j})"));
    }
    let bytes = max_duration
        .iter()
        .try_fold(0usize, |acc, &m| acc.checked_add(m))
        .and_then(|s| s.checked_mul(DURATION_TABLES * size_of::<f64>()));
    match bytes {
        Some(b) => memory::check(b as u64, None, "duration tables"),
        None => invalid("max_duration values overflow usize"),
    }
}

impl Hsmm {
    pub fn new(
        start: Vec<f64>,
        transition: Mat,
        durations: Vec<DurationDist>,
        emissions: Vec<Emission>,
        max_duration: Vec<usize>,
    ) -> Result<Self> {
        let model = Hsmm {
            start,
            transition,
            durations,
            emissions,
            max_duration,
        };
        model.validate()?;
        Ok(model)
    }

    /// Public fields and deserialised JSON bypass `new`, so entry points validate again.
    pub(crate) fn validate(&self) -> Result<()> {
        let Hsmm {
            start,
            transition,
            durations,
            emissions,
            max_duration,
        } = self;
        let k = start.len();
        if k == 0 {
            return invalid("an HSMM needs at least one state");
        }
        if transition.nrows() != k || transition.ncols() != k {
            return invalid(format!("transition must be {k} x {k}"));
        }
        if durations.len() != k || emissions.len() != k || max_duration.len() != k {
            return invalid(format!(
                "give {k} durations, emissions and max_duration, not {}, {} and {}",
                durations.len(),
                emissions.len(),
                max_duration.len()
            ));
        }
        check_max_duration(max_duration)?;
        check_probabilities(start, "start")?;
        for i in 0..k {
            if transition[(i, i)] != 0.0 {
                return invalid(format!("transition diagonal must be 0 (row {i})"));
            }
            if k > 1 {
                let row: Vec<f64> = transition.row(i).iter().copied().collect();
                check_probabilities(&row, &format!("transition row {i}"))?;
            }
        }
        for (j, (d, &m)) in durations.iter().zip(max_duration).enumerate() {
            if !(d.pseudocount >= 0.0 && d.pseudocount.is_finite()) {
                return invalid(format!("duration {j} pseudocount must be non-negative"));
            }
            d.law.validate(m)?;
        }
        for e in emissions {
            e.validate()?;
        }
        let dim = emissions[0].dim();
        if emissions.iter().any(|e| e.dim() != dim) {
            return invalid("every emission must have the same number of channels");
        }
        Ok(())
    }

    /// Geometric equivalence: `d_j` geometric with `p = 1 - a_jj`, `p_ij = a_ij / (1 - a_jj)`.
    pub fn from_hmm(hmm: &Hmm, max_duration: usize) -> Result<Self> {
        let k = hmm.start.len();
        let a = &hmm.transition;
        let mut transition = Mat::zeros(k, k);
        let mut durations = Vec::with_capacity(k);
        for j in 0..k {
            let stay = a[(j, j)];
            if stay >= 1.0 {
                return invalid(format!(
                    "state {j} is absorbing (a_jj = 1): no geometric HSMM equivalent"
                ));
            }
            for i in 0..k {
                if i != j {
                    transition[(j, i)] = a[(j, i)] / (1.0 - stay);
                }
            }
            durations.push(DurationDist {
                law: Duration::Geometric { p: 1.0 - stay },
                frozen: false,
                pseudocount: 0.0,
            });
        }
        Hsmm::new(
            hmm.start.clone(),
            transition,
            durations,
            hmm.emissions.clone(),
            vec![max_duration; k],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        k: usize,
        family: &Family,
        duration: DurationFamily,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        priors: &Priors,
        max_duration: &[usize],
        stream: &mut Stream,
    ) -> Result<Self> {
        if max_duration.len() != k {
            return invalid(format!(
                "give {k} max_duration values, not {}",
                max_duration.len()
            ));
        }
        check_max_duration(max_duration)?;
        if clamps.is_some_and(|c| c.len() != data.len()) {
            return invalid("give one clamp per sequence");
        }
        let hmm = Hmm::initialize(k, family, data, clamps, priors, stream)?;
        let mut complete: Vec<Vec<f64>> = max_duration.iter().map(|&m| vec![0.0; m]).collect();
        let mut censored = complete.clone();
        let mut trans = Mat::from_fn(k, k, |i, j| if i == j { 0.0 } else { 1.0 });
        let mut first = vec![1.0; k];
        for (n, seq) in data.iter().enumerate() {
            let (_, lb) = log_b(&hmm.emissions, seq, clamps.map(|c| &c[n]))?;
            let labels: Vec<usize> = (0..seq.t)
                .map(|t| {
                    let mut best = 0;
                    for j in 1..k {
                        if lb[(t, j)] > lb[(t, best)] {
                            best = j;
                        }
                    }
                    best
                })
                .collect();
            if labels.is_empty() {
                continue;
            }
            first[labels[0]] += 1.0;
            let mut t0 = 0;
            for t in 1..=labels.len() {
                if t < labels.len() && labels[t] == labels[t0] {
                    continue;
                }
                let s = labels[t0];
                let len = (t - t0).min(max_duration[s]);
                if t < labels.len() {
                    complete[s][len - 1] += 1.0;
                    trans[(s, labels[t])] += 1.0;
                } else {
                    censored[s][len - 1] += 1.0;
                }
                t0 = t;
            }
        }
        let pooled = |j: usize| -> Vec<f64> {
            let m = max_duration[j];
            let mut out = vec![0.0; m];
            for (c, z) in complete.iter().zip(&censored) {
                for (u, (a, b)) in c.iter().zip(z).enumerate() {
                    out[u.min(m - 1)] += a + b;
                }
            }
            out
        };
        let mut durations = Vec::with_capacity(k);
        for j in 0..k {
            let m = max_duration[j];
            let own_all: Vec<f64> = complete[j]
                .iter()
                .zip(&censored[j])
                .map(|(a, b)| a + b)
                .collect();
            let counts = [complete[j].clone(), own_all, pooled(j)]
                .into_iter()
                .find(|c| c.iter().sum::<f64>() > 0.0)
                .unwrap_or_else(|| {
                    let mut c = vec![0.0; m];
                    c[0] = 1.0;
                    c
                });
            let pc = if duration == DurationFamily::Empirical {
                1.0
            } else {
                0.0
            };
            let law = duration.template(m).fit(&counts, pc)?;
            durations.push(DurationDist {
                law,
                frozen: false,
                pseudocount: 0.0,
            });
        }
        let total: f64 = first.iter().sum();
        let start = first.iter().map(|v| v / total).collect();
        if k > 1 {
            for i in 0..k {
                let s: f64 = trans.row(i).sum();
                for j in 0..k {
                    trans[(i, j)] /= s;
                }
            }
        }
        Hsmm::new(
            start,
            trans,
            durations,
            hmm.emissions,
            max_duration.to_vec(),
        )
    }

    pub(crate) fn k(&self) -> usize {
        self.start.len()
    }

    pub(crate) fn tables(&self) -> Tables {
        let laws: Vec<&Duration> = self.durations.iter().map(|d| &d.law).collect();
        Tables::new(&self.start, &self.transition, &laws, &self.max_duration)
    }

    pub fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)> {
        let tables = self.tables();
        let k = self.k();
        let d = self.emissions[0].dim();
        lengths
            .par_iter()
            .enumerate()
            .map(|(i, &n)| {
                let mut stream = Stream::new(seed, &[tag::SAMPLE, i as u64]);
                let mut states = Vec::with_capacity(n);
                let mut x = Vec::with_capacity(n * d);
                if n == 0 {
                    return (states, Sequence { t: 0, d, x });
                }
                let mut s = stream.categorical(&self.start);
                loop {
                    let u = Duration::draw(&tables.log_d[s], &mut stream);
                    for _ in 0..u.min(n - states.len()) {
                        states.push(s as u32);
                        x.extend(self.emissions[s].sample(&mut stream));
                    }
                    if states.len() >= n {
                        break;
                    }
                    if k > 1 {
                        let row: Vec<f64> = self.transition.row(s).iter().copied().collect();
                        s = stream.categorical(&row);
                    }
                }
                (states, Sequence { t: n, d, x })
            })
            .collect()
    }

    pub fn n_params(&self, params: Params) -> usize {
        let k = self.k();
        let mut n = 0;
        if params.start {
            n += self
                .start
                .iter()
                .filter(|p| **p > 0.0)
                .count()
                .saturating_sub(1);
        }
        if params.transition && k > 2 {
            for i in 0..k {
                n += self
                    .transition
                    .row(i)
                    .iter()
                    .filter(|p| **p > 0.0)
                    .count()
                    .saturating_sub(1);
            }
        }
        if params.duration {
            n += self
                .durations
                .iter()
                .zip(&self.max_duration)
                .filter(|(d, _)| !d.frozen)
                .map(|(d, &m)| d.law.n_params(m))
                .sum::<usize>();
        }
        if params.emission {
            n += self.emissions.iter().map(Emission::n_params).sum::<usize>();
        }
        n
    }

    /// Untruncated law's mass above `max_duration`; raise `max_duration` when it exceeds `1e-6`.
    pub fn truncated_mass(&self) -> Vec<f64> {
        self.durations
            .iter()
            .zip(&self.max_duration)
            .map(|(d, &m)| d.law.mass_above(m))
            .collect()
    }
}
