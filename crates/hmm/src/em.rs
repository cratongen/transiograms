use std::collections::HashMap;

use rayon::prelude::*;
use transiograms_core::nalgebra::Cholesky;
use transiograms_core::numerics::{gth_stationary, logsumexp};
use transiograms_core::rng::tag;
use transiograms_core::{Mat, Result, Stream, convergence, invalid};

use crate::emission::{
    Categorical, CovarianceType, Emission, Family, GaussDensity, Gaussian, GaussianMixture,
    Poisson, chol_logdet, observed,
};
use crate::forward_backward::Pass;
use crate::init::{check_data, initial_emissions, label_counts};
use crate::model::{FitOptions, FitReport, Hmm, Ordering, n_observed, validate_emissions};
use crate::sequence::{Clamp, Sequence, pooled_moments};

impl Hmm {
    /// Restarts are fit in parallel; restart `r` uses `Stream::new(seed, &[tag::INIT, r])`.
    pub fn fit(
        &mut self,
        family: &Family,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
    ) -> Result<FitReport> {
        opts.priors.validate()?;
        if !(opts.tol.is_finite() && opts.tol >= 0.0) {
            return invalid(format!(
                "tol must be finite and non-negative, not {}",
                opts.tol
            ));
        }
        let k = self.k();
        check_data(k, family, data, clamps)?;
        if opts.ordering == Ordering::Mean && matches!(family, Family::Categorical { .. }) {
            return invalid("ordering by mean needs continuous emissions; use occupancy");
        }
        let (_, v) = pooled_moments(data);
        let n_obs = n_observed(data).max(1) as f64;
        let ip = opts.init_params;
        let restarts = if ip.start || ip.transition || ip.emission {
            opts.n_init.max(1)
        } else {
            1
        };
        let results: Vec<Result<(Hmm, FitReport)>> = (0..restarts)
            .into_par_iter()
            .map(|r| {
                let mut stream = Stream::new(opts.seed, &[tag::INIT, r as u64]);
                let model = self.restart_model(family, data, clamps, opts, &v, &mut stream)?;
                let (model, mut report) = model.em(data, clamps, opts, &v, n_obs)?;
                report.restart = r;
                Ok((model, report))
            })
            .collect();
        let mut best: Option<(Hmm, FitReport)> = None;
        for res in results {
            let (m, rep) = res?;
            let obj = *rep.history.last().unwrap_or(&f64::NEG_INFINITY);
            let better = best
                .as_ref()
                .is_none_or(|(_, b)| obj > *b.history.last().unwrap_or(&f64::NEG_INFINITY));
            if better {
                best = Some((m, rep));
            }
        }
        let (mut model, report) = best.expect("at least one restart");
        model.reorder(opts.ordering)?;
        *self = model;
        Ok(report)
    }

    fn restart_model(
        &self,
        family: &Family,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
        v: &[f64],
        stream: &mut Stream,
    ) -> Result<Hmm> {
        let k = self.k();
        let ip = opts.init_params;
        let frozen = |j: usize| self.emissions.get(j).is_some_and(Emission::frozen);
        let emissions = if ip.emission && !(0..k).all(frozen) {
            let fresh =
                initial_emissions(k, family, data, clamps, opts.priors.covariance, v, stream)?;
            fresh
                .into_iter()
                .enumerate()
                .map(|(j, e)| match (e, self.emissions.get(j)) {
                    _ if frozen(j) => self.emissions[j].clone(),
                    (Emission::Categorical(mut c), Some(Emission::Categorical(t))) => {
                        c.pseudocount = t.pseudocount;
                        Emission::Categorical(c)
                    }
                    (e, _) => e,
                })
                .collect()
        } else {
            self.emissions.clone()
        };
        validate_emissions(&emissions, k)?;
        let (mut start, mut transition) = (self.start.clone(), self.transition.clone());
        if ip.start || ip.transition {
            let (s, t) = label_counts(&emissions, data, clamps, k);
            if ip.start {
                start = s;
            }
            if ip.transition {
                transition = t;
            }
        }
        let model = Hmm::new(start, transition, emissions)?;
        for (n, seq) in data.iter().enumerate() {
            model.check_sequence(seq, clamps.map(|c| &c[n]), n)?;
        }
        Ok(model)
    }

    fn em(
        mut self,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
        v: &[f64],
        n_obs: f64,
    ) -> Result<(Hmm, FitReport)> {
        let mut history: Vec<f64> = Vec::new();
        let mut converged = false;
        let mut m_steps = 0;
        loop {
            let stats: Vec<Pass> = data
                .par_iter()
                .enumerate()
                .map(|(n, seq)| {
                    let lb = self.emission_logs(seq, clamps.map(|c| &c[n]));
                    self.forward_backward(&lb, opts.implementation, n)
                })
                .collect::<Result<_>>()?;
            let obj = stats.iter().map(|s| s.ll).sum::<f64>() + self.log_prior(&opts.priors, v);
            if !obj.is_finite() {
                return convergence(format!(
                    "objective is {obj} at iteration {m_steps}; the model went non-finite"
                ));
            }
            let prev = history.last().copied();
            history.push(obj);
            if let Some(prev) = prev {
                if obj < prev - 1e-9 * obj.abs() {
                    return convergence(format!(
                        "objective decreased at iteration {m_steps}: {prev} to {obj}"
                    ));
                }
                if (obj - prev) / n_obs < opts.tol {
                    converged = true;
                    break;
                }
            }
            if m_steps == opts.max_iter {
                break;
            }
            self.m_step(data, &stats, opts, v)?;
            m_steps += 1;
        }
        Ok((
            self,
            FitReport {
                history,
                converged,
                n_iter: m_steps,
                restart: 0,
            },
        ))
    }

    fn m_step(
        &mut self,
        data: &[Sequence],
        stats: &[Pass],
        opts: &FitOptions,
        v: &[f64],
    ) -> Result<()> {
        let k = self.k();
        let (p, pr) = (opts.params, opts.priors);
        if p.start {
            let mut num = vec![0.0; k];
            for s in stats {
                for j in 0..k {
                    num[j] += s.gamma[(0, j)];
                }
            }
            for j in 0..k {
                num[j] = if self.start[j] > 0.0 {
                    num[j] + pr.start - 1.0
                } else {
                    0.0
                };
            }
            let total: f64 = num.iter().sum();
            if total > 0.0 {
                self.start = num.iter().map(|x| x / total).collect();
            }
        }
        if p.transition {
            let mut xi = Mat::zeros(k, k);
            for s in stats {
                xi += &s.xi_sum;
            }
            for i in 0..k {
                let occupancy: f64 = (0..k)
                    .filter(|&j| self.transition[(i, j)] > 0.0)
                    .map(|j| xi[(i, j)])
                    .sum();
                if occupancy <= 0.0 {
                    continue;
                }
                let num: Vec<f64> = (0..k)
                    .map(|j| {
                        if self.transition[(i, j)] > 0.0 {
                            xi[(i, j)] + pr.transition - 1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let total: f64 = num.iter().sum();
                for j in 0..k {
                    self.transition[(i, j)] = num[j] / total;
                }
            }
        }
        if p.emission {
            let eps = pr.covariance;
            let updated: Vec<Result<Emission>> = (0..k)
                .into_par_iter()
                .map(|j| {
                    let w: Vec<Vec<f64>> = stats
                        .iter()
                        .map(|s| s.gamma.column(j).iter().copied().collect())
                        .collect();
                    update_emission(&self.emissions[j], j, data, &w, eps, v)
                })
                .collect();
            self.emissions = updated.into_iter().collect::<Result<_>>()?;
        }
        Ok(())
    }

    fn reorder(&mut self, ordering: Ordering) -> Result<()> {
        let k = self.k();
        let key: Vec<f64> = match ordering {
            Ordering::None => return Ok(()),
            Ordering::Mean => self
                .emissions
                .iter()
                .map(|e| match e {
                    Emission::Gaussian(g) => Ok(g.mean[0]),
                    Emission::Mixture(m) => Ok(m
                        .weights
                        .iter()
                        .zip(&m.components)
                        .map(|(w, g)| w * g.mean[0])
                        .sum()),
                    Emission::Poisson(p) => Ok(p.rate[0]),
                    Emission::Categorical(_) => {
                        invalid("ordering by mean needs continuous emissions; use occupancy")
                    }
                })
                .collect::<Result<_>>()?,
            Ordering::Occupancy => gth_stationary(&self.transition)?
                .iter()
                .map(|p| -p)
                .collect(),
        };
        let mut perm: Vec<usize> = (0..k).collect();
        perm.sort_by(|&a, &b| key[a].total_cmp(&key[b]));
        self.start = perm.iter().map(|&i| self.start[i]).collect();
        self.transition = Mat::from_fn(k, k, |i, j| self.transition[(perm[i], perm[j])]);
        self.emissions = perm.iter().map(|&i| self.emissions[i].clone()).collect();
        Ok(())
    }
}

/// Weighted M-step of one state's emission, `w[n][t]` the occupancy of sequence `n` at step `t`;
/// shared with the HSMM.
pub fn update_emission(
    e: &Emission,
    state: usize,
    data: &[Sequence],
    w: &[Vec<f64>],
    eps: f64,
    v: &[f64],
) -> Result<Emission> {
    if e.frozen() {
        return Ok(e.clone());
    }
    Ok(match e {
        Emission::Categorical(c) => {
            let m = c.probabilities.len();
            let mut num = vec![c.pseudocount; m];
            for (seq, ws) in data.iter().zip(w) {
                for t in 0..seq.t {
                    let x = seq.x[t];
                    if !x.is_nan() {
                        num[x as usize] += ws[t];
                    }
                }
            }
            let total: f64 = num.iter().sum();
            if total > 0.0 {
                Emission::Categorical(Categorical {
                    probabilities: num.iter().map(|x| x / total).collect(),
                    ..c.clone()
                })
            } else {
                e.clone()
            }
        }
        Emission::Poisson(p) => {
            let d = p.rate.len();
            let (mut g, mut s) = (vec![0.0; d], vec![0.0; d]);
            for (seq, ws) in data.iter().zip(w) {
                for t in 0..seq.t {
                    for (c, &x) in seq.row(t).iter().enumerate() {
                        if !x.is_nan() {
                            g[c] += ws[t];
                            s[c] += ws[t] * x;
                        }
                    }
                }
            }
            Emission::Poisson(Poisson {
                rate: (0..d)
                    .map(|c| if g[c] > 0.0 { s[c] / g[c] } else { p.rate[c] })
                    .collect(),
                frozen: false,
            })
        }
        Emission::Gaussian(g) => Emission::Gaussian(update_gaussian(g, data, w, eps, v, state)?),
        Emission::Mixture(mix) => {
            let nm = mix.components.len();
            let mut dens: Vec<(f64, GaussDensity)> = mix
                .weights
                .iter()
                .zip(&mix.components)
                .map(|(wt, g)| (wt.ln(), GaussDensity::new(g)))
                .collect();
            let mut r: Vec<Vec<Vec<f64>>> = vec![Vec::with_capacity(data.len()); nm];
            let mut terms = vec![0.0; nm];
            for (seq, ws) in data.iter().zip(w) {
                let mut rs = vec![vec![0.0; seq.t]; nm];
                for t in 0..seq.t {
                    let x = seq.row(t);
                    if ws[t] <= 0.0 || x.iter().all(|v| v.is_nan()) {
                        continue;
                    }
                    for (m, (lw, g)) in dens.iter_mut().enumerate() {
                        terms[m] = *lw + g.log_density(x);
                    }
                    let lb = logsumexp(&terms);
                    if lb == f64::NEG_INFINITY {
                        continue;
                    }
                    for m in 0..nm {
                        rs[m][t] = ws[t] * (terms[m] - lb).exp();
                    }
                }
                for (m, rm) in rs.into_iter().enumerate() {
                    r[m].push(rm);
                }
            }
            let mass: Vec<f64> = r
                .iter()
                .map(|rm| rm.iter().flatten().sum::<f64>())
                .collect();
            let total: f64 = mass.iter().sum();
            if total <= 0.0 {
                return Ok(e.clone());
            }
            let components = mix
                .components
                .iter()
                .zip(&r)
                .map(|(g, rm)| update_gaussian(g, data, rm, eps, v, state))
                .collect::<Result<Vec<_>>>()?;
            Emission::Mixture(GaussianMixture {
                weights: mass.iter().map(|m| m / total).collect(),
                components,
                frozen: false,
            })
        }
    })
}

/// Conditional moments of the missing block given the observed one, per missingness pattern:
/// `K = S_mo S_oo^-1` and `Cov = S_mm - K S_om`.
type CondMoments = Option<(Vec<usize>, Vec<usize>, Mat, Mat)>;

fn conditional(s: &Mat, obs: &[usize]) -> CondMoments {
    let d = s.nrows();
    let mis: Vec<usize> = (0..d).filter(|c| !obs.contains(c)).collect();
    let soo = s.select_rows(obs).select_columns(obs);
    let smo = s.select_rows(&mis).select_columns(obs);
    let smm = s.select_rows(&mis).select_columns(&mis);
    let chol = Cholesky::new(soo)?;
    let kmo = chol.solve(&smo.transpose()).transpose();
    let cov = &smm - &kmo * smo.transpose();
    Some((obs.to_vec(), mis, kmo, cov))
}

fn update_gaussian(
    old: &Gaussian,
    data: &[Sequence],
    w: &[Vec<f64>],
    eps: f64,
    v: &[f64],
    state: usize,
) -> Result<Gaussian> {
    let d = old.mean.len();
    let vc = |c: usize| v.get(c).copied().unwrap_or(0.0);
    let (mean, mut s) = match old.covariance_type {
        CovarianceType::Full => {
            let mut wsum = 0.0;
            let mut mean = vec![0.0; d];
            let mut c = Mat::zeros(d, d);
            let mut cache: HashMap<Vec<usize>, CondMoments> = HashMap::new();
            let mut xhat = vec![0.0; d];
            let mut delta = vec![0.0; d];
            for (seq, ws) in data.iter().zip(w) {
                for t in 0..seq.t {
                    let wt = ws[t];
                    let x = seq.row(t);
                    if wt <= 0.0 {
                        continue;
                    }
                    let obs = observed(x);
                    if obs.is_empty() {
                        continue;
                    }
                    xhat.copy_from_slice(x);
                    if obs.len() < d {
                        let entry = cache
                            .entry(obs.clone())
                            .or_insert_with(|| conditional(&old.covariance, &obs));
                        let Some((o, m, kmo, cm)) = entry.as_ref() else {
                            return convergence(format!(
                                "covariance of state {state} is not positive definite"
                            ));
                        };
                        for (a, &ma) in m.iter().enumerate() {
                            xhat[ma] = old.mean[ma]
                                + o.iter()
                                    .enumerate()
                                    .map(|(b, &ob)| kmo[(a, b)] * (x[ob] - old.mean[ob]))
                                    .sum::<f64>();
                            for (b, &mb) in m.iter().enumerate() {
                                c[(ma, mb)] += wt * cm[(a, b)];
                            }
                        }
                    }
                    // Weighted West update: C += w (x - mean_old)(x - mean_new)^T, written
                    // as w (1 - w / W) delta delta^T to stay exactly symmetric.
                    wsum += wt;
                    let f = wt * (1.0 - wt / wsum);
                    for a in 0..d {
                        delta[a] = xhat[a] - mean[a];
                        mean[a] += wt / wsum * delta[a];
                    }
                    for a in 0..d {
                        for b in 0..d {
                            c[(a, b)] += f * delta[a] * delta[b];
                        }
                    }
                }
            }
            if wsum <= 0.0 {
                return Ok(old.clone());
            }
            let s = if eps > 0.0 {
                for a in 0..d {
                    c[(a, a)] += eps * vc(a);
                }
                c / (wsum + 1.0)
            } else {
                c / wsum
            };
            let s = (&s + s.transpose()) * 0.5;
            (mean, s)
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            let (mut g, mut mean, mut c) = (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
            for (seq, ws) in data.iter().zip(w) {
                for t in 0..seq.t {
                    let wt = ws[t];
                    if wt <= 0.0 {
                        continue;
                    }
                    for (a, &x) in seq.row(t).iter().enumerate() {
                        if x.is_nan() {
                            continue;
                        }
                        g[a] += wt;
                        let delta = x - mean[a];
                        mean[a] += wt / g[a] * delta;
                        c[a] += wt * (1.0 - wt / g[a]) * delta * delta;
                    }
                }
            }
            if g.iter().all(|x| *x <= 0.0) {
                return Ok(old.clone());
            }
            for a in 0..d {
                if g[a] <= 0.0 {
                    mean[a] = old.mean[a];
                }
            }
            let s = if old.covariance_type == CovarianceType::Diag {
                Mat::from_fn(d, d, |a, b| {
                    if a != b {
                        0.0
                    } else if g[a] <= 0.0 {
                        old.covariance[(a, a)]
                    } else if eps > 0.0 {
                        (c[a] + eps * vc(a)) / (g[a] + 1.0)
                    } else {
                        c[a] / g[a]
                    }
                })
            } else {
                let (sc, sg): (f64, f64) = (c.iter().sum(), g.iter().sum());
                let s2 = if eps > 0.0 {
                    (sc + eps * (0..d).map(vc).sum::<f64>()) / (sg + d as f64)
                } else {
                    sc / sg
                };
                Mat::identity(d, d) * s2
            };
            (mean, s)
        }
    };
    if chol_logdet(&s).is_none() {
        let jitter = 1e-10 * s.trace() / d as f64;
        for a in 0..d {
            s[(a, a)] += jitter;
        }
        if jitter.is_nan() || jitter <= 0.0 || chol_logdet(&s).is_none() {
            return convergence(format!(
                "covariance of state {state} is not positive definite after jitter"
            ));
        }
    }
    Ok(Gaussian {
        mean,
        covariance: s,
        covariance_type: old.covariance_type,
        frozen: false,
    })
}
