use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use varvite_core::rng::tag;
use varvite_core::{Result, Stream, invalid};
use varvite_hmm::{Clamp, Family, FitOptions, Hmm, Implementation, Params, Sequence};

use crate::duration::DurationFamily;
use crate::model::Hsmm;
use varvite_hmm::n_observed;

#[derive(Clone, Debug, PartialEq)]
pub enum Model {
    Hmm(Hmm),
    Hsmm(Hsmm),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Hmm,
    Hsmm {
        duration: DurationFamily,
        max_duration: Option<usize>,
    },
}

/// Depth samples are autocorrelated, so BIC tends to over-select K on dense logs; the
/// cross-validated score is the honest check.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionSpec {
    pub states: Vec<usize>,
    pub family: Family,
    pub kind: ModelKind,
    pub fit: FitOptions,
    /// None = leave one hole out, Some(0) = no CV.
    pub cv_folds: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionTable {
    pub states: Vec<usize>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub log_likelihood: Vec<f64>,
    pub n_params: Vec<usize>,
    pub n_observations: usize,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub aic: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub bic: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub icl: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub cv_log_likelihood: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub cv_per_step: Vec<f64>,
}

pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * n_params as f64
}

pub fn bic(log_likelihood: f64, n_params: usize, n_observations: usize) -> f64 {
    -2.0 * log_likelihood + n_params as f64 * (n_observations as f64).ln()
}

/// `entropy` is the exact path entropy `H(S | X)` summed over sequences.
pub fn icl(log_likelihood: f64, n_params: usize, n_observations: usize, entropy: f64) -> f64 {
    bic(log_likelihood, n_params, n_observations) + 2.0 * entropy
}

impl Model {
    fn score(&self, seq: &Sequence) -> Result<f64> {
        match self {
            Model::Hsmm(m) => m.log_likelihood_or_neg_inf(seq, None),
            Model::Hmm(m) => match m.log_likelihood(seq, None, Implementation::Log) {
                Ok(v) => Ok(v),
                Err(varvite_core::Error::InvalidInput(msg)) if msg.contains("impossible") => {
                    Ok(f64::NEG_INFINITY)
                }
                Err(e) => Err(e),
            },
        }
    }

    fn entropy(&self, seq: &Sequence) -> Result<f64> {
        match self {
            Model::Hsmm(m) => Ok(m.posterior(seq, None)?.entropy),
            Model::Hmm(m) => Ok(m.posterior(seq, None, Implementation::Log)?.entropy),
        }
    }

    fn n_params(&self, params: Params) -> usize {
        match self {
            Model::Hsmm(m) => m.n_params(params),
            Model::Hmm(m) => m.n_params(params),
        }
    }
}

/// Restart `r` uses `Stream::new(spec.fit.seed, keys ++ [r])`.
fn fit_one(
    k: usize,
    data: &[Sequence],
    clamps: Option<&[Clamp]>,
    spec: &SelectionSpec,
    max_duration: usize,
    keys: &[u64],
) -> Result<Model> {
    match spec.kind {
        ModelKind::Hsmm { duration, .. } => {
            let all = Params {
                start: true,
                transition: true,
                emission: true,
                duration: true,
            };
            let opts = FitOptions {
                init_params: all,
                ..spec.fit.clone()
            };
            let (m, _) = Hsmm::fit_restarts(
                None,
                k,
                &spec.family,
                duration,
                &vec![max_duration; k],
                data,
                clamps,
                &opts,
                keys,
            )?;
            Ok(Model::Hsmm(m))
        }
        ModelKind::Hmm => {
            let none = Params {
                start: false,
                transition: false,
                emission: false,
                duration: false,
            };
            let opts = FitOptions {
                n_init: 1,
                init_params: none,
                ..spec.fit.clone()
            };
            let fits: Vec<Result<(Hmm, f64)>> = (0..spec.fit.n_init.max(1))
                .into_par_iter()
                .map(|r| {
                    let mut all = keys.to_vec();
                    all.push(r as u64);
                    let mut stream = Stream::new(spec.fit.seed, &all);
                    let mut hmm = Hmm::initialize(
                        k,
                        &spec.family,
                        data,
                        clamps,
                        &spec.fit.priors,
                        &mut stream,
                    )?;
                    let report = hmm.fit(&spec.family, data, clamps, &opts)?;
                    let last = report.history.last().copied().unwrap_or(f64::NEG_INFINITY);
                    Ok((hmm, last))
                })
                .collect();
            let mut best: Option<(Hmm, f64)> = None;
            for f in fits {
                let (h, obj) = f?;
                if best.as_ref().is_none_or(|b| obj > b.1) {
                    best = Some((h, obj));
                }
            }
            Ok(Model::Hmm(best.expect("at least one restart").0))
        }
    }
}

pub fn select_states(
    data: &[Sequence],
    clamps: Option<&[Clamp]>,
    spec: &SelectionSpec,
) -> Result<SelectionTable> {
    if data.is_empty() {
        return invalid("select_states needs at least one sequence");
    }
    if spec.states.is_empty() || spec.states.contains(&0) {
        return invalid("states must be a non-empty list of positive state counts");
    }
    if clamps.is_some_and(|c| c.len() != data.len()) {
        return invalid("give one clamp per sequence");
    }
    let n_holes = data.len();
    let max_duration = match spec.kind {
        ModelKind::Hsmm {
            max_duration: Some(m),
            ..
        } => m,
        _ => data.iter().map(|s| s.t).max().unwrap_or(1).max(1),
    };
    let fold_of: Vec<usize> = match spec.cv_folds {
        Some(0) => Vec::new(),
        None => (0..n_holes).collect(),
        Some(k) => {
            let mut perm: Vec<usize> = (0..n_holes).collect();
            Stream::new(spec.fit.seed, &[tag::FOLD]).shuffle(&mut perm);
            let mut fold = vec![0; n_holes];
            for (pos, &h) in perm.iter().enumerate() {
                fold[h] = pos % k;
            }
            fold
        }
    };
    let n_folds = match spec.cv_folds {
        Some(0) => 0,
        None => n_holes,
        Some(k) => k,
    };
    if n_folds > 0 && n_holes < 2 {
        return invalid("cross-validation needs at least two sequences; use cv_folds = Some(0)");
    }
    let n_states = spec.states.len();
    let jobs: Vec<(usize, Option<usize>)> = (0..n_states)
        .flat_map(|s| std::iter::once((s, None)).chain((0..n_folds).map(move |f| (s, Some(f)))))
        .collect();
    let results: Vec<Result<(f64, f64, usize)>> = jobs
        .par_iter()
        .map(|&(s, fold)| {
            let k = spec.states[s];
            match fold {
                None => {
                    let model =
                        fit_one(k, data, clamps, spec, max_duration, &[tag::INIT, s as u64])?;
                    let mut lnl = 0.0;
                    let mut ent = 0.0;
                    for seq in data {
                        lnl += model.score(seq)?;
                        ent += model.entropy(seq)?;
                    }
                    Ok((lnl, ent, model.n_params(spec.fit.params)))
                }
                Some(f) => {
                    let held: Vec<usize> = (0..n_holes).filter(|&h| fold_of[h] == f).collect();
                    if held.is_empty() {
                        return Ok((0.0, 0.0, 0));
                    }
                    let train_idx: Vec<usize> = (0..n_holes).filter(|&h| fold_of[h] != f).collect();
                    if train_idx.is_empty() {
                        return invalid(format!("fold {f} leaves no sequence to train on"));
                    }
                    let train: Vec<Sequence> = train_idx.iter().map(|&h| data[h].clone()).collect();
                    let train_clamps: Option<Vec<Clamp>> =
                        clamps.map(|c| train_idx.iter().map(|&h| c[h].clone()).collect());
                    let model = fit_one(
                        k,
                        &train,
                        train_clamps.as_deref(),
                        spec,
                        max_duration,
                        &[tag::FOLD, f as u64, s as u64],
                    )?;
                    let mut lnl = 0.0;
                    for &h in &held {
                        lnl += model.score(&data[h])?;
                    }
                    Ok((lnl, 0.0, 0))
                }
            }
        })
        .collect();
    let n = n_observed(data);
    let mut table = SelectionTable {
        states: spec.states.clone(),
        log_likelihood: Vec::with_capacity(n_states),
        n_params: Vec::with_capacity(n_states),
        n_observations: n,
        aic: Vec::with_capacity(n_states),
        bic: Vec::with_capacity(n_states),
        icl: Vec::with_capacity(n_states),
        cv_log_likelihood: Vec::with_capacity(n_states),
        cv_per_step: Vec::with_capacity(n_states),
    };
    let mut it = results.into_iter();
    for _ in 0..n_states {
        let (lnl, ent, np) = it.next().expect("job per state")?;
        let mut cv = 0.0;
        for _ in 0..n_folds {
            cv += it.next().expect("job per fold")?.0;
        }
        if n_folds == 0 {
            cv = f64::NAN;
        }
        table.log_likelihood.push(lnl);
        table.n_params.push(np);
        table.aic.push(aic(lnl, np));
        table.bic.push(bic(lnl, np, n));
        table.icl.push(icl(lnl, np, n, ent));
        table.cv_log_likelihood.push(cv);
        table.cv_per_step.push(cv / n as f64);
    }
    Ok(table)
}
