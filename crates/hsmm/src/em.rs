use rayon::prelude::*;
use transiograms_core::numerics::gth_stationary;
use transiograms_core::rng::tag;
use transiograms_core::{Mat, Result, Stream, convergence, invalid};
use transiograms_hmm::{Clamp, Emission, Family, FitOptions, FitReport, Ordering, Sequence};

use crate::duration::{Duration, DurationFamily};
use crate::forward_backward::{SegmentPosterior, log_b, posterior_core};
use crate::model::Hsmm;
use transiograms_hmm::em::update_emission;
use transiograms_hmm::model::emission_log_prior;
use transiograms_hmm::n_observed;
use transiograms_hmm::sequence::pooled_moments;

fn sum_xlogy(xs: impl Iterator<Item = f64>, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    xs.filter(|p| *p > 0.0).map(|p| a * p.ln()).sum()
}

impl Hsmm {
    fn log_prior(&self, opts: &FitOptions, v: &[f64]) -> f64 {
        let k = self.k();
        let pr = &opts.priors;
        let mut lp = sum_xlogy(self.start.iter().copied(), pr.start - 1.0);
        lp += sum_xlogy(
            (0..k)
                .flat_map(|i| (0..k).filter(move |&j| j != i).map(move |j| (i, j)))
                .map(|ij| self.transition[ij]),
            pr.transition - 1.0,
        );
        for (d, &m) in self.durations.iter().zip(&self.max_duration) {
            if let Duration::Empirical { .. } = d.law {
                lp += sum_xlogy(d.law.log_pmf(m).iter().map(|l| l.exp()), d.pseudocount);
            }
        }
        lp + self
            .emissions
            .iter()
            .map(|e| emission_log_prior(e, pr.covariance, v))
            .sum::<f64>()
    }

    fn e_step(&self, data: &[Sequence], clamps: Option<&[Clamp]>) -> Result<Vec<SegmentPosterior>> {
        let tables = self.tables();
        let posts: Vec<Result<SegmentPosterior>> = data
            .par_iter()
            .enumerate()
            .map(|(n, seq)| {
                let (free, lb) = log_b(&self.emissions, seq, clamps.map(|c| &c[n]))?;
                posterior_core(&tables, &lb, &free, Some(n))
            })
            .collect();
        posts.into_iter().collect()
    }

    fn m_step(
        &mut self,
        data: &[Sequence],
        posts: &[SegmentPosterior],
        opts: &FitOptions,
        v: &[f64],
    ) -> Result<()> {
        let k = self.k();
        let pr = &opts.priors;
        if opts.params.start {
            let num: Vec<f64> = (0..k)
                .map(|j| {
                    if self.start[j] > 0.0 {
                        (posts.iter().map(|p| p.start[j]).sum::<f64>() + pr.start - 1.0).max(0.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f64 = num.iter().sum();
            if total > 0.0 {
                self.start = num.iter().map(|x| x / total).collect();
            }
        }
        if opts.params.transition && k > 1 {
            let mut xi = Mat::zeros(k, k);
            for p in posts {
                xi += &p.xi_sum;
            }
            for i in 0..k {
                let num: Vec<f64> = (0..k)
                    .map(|j| {
                        if j != i && self.transition[(i, j)] > 0.0 {
                            (xi[(i, j)] + pr.transition - 1.0).max(0.0)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let total: f64 = num.iter().sum();
                if total > 0.0 {
                    for j in 0..k {
                        self.transition[(i, j)] = num[j] / total;
                    }
                }
            }
        }
        if opts.params.duration {
            for j in 0..k {
                if self.durations[j].frozen {
                    continue;
                }
                let mut counts = vec![0.0; self.max_duration[j]];
                for p in posts {
                    for (c, v) in counts.iter_mut().zip(&p.durations[j]) {
                        *c += v;
                    }
                }
                if counts.iter().sum::<f64>() > 0.0 {
                    let pc = self.durations[j].pseudocount;
                    let m = self.max_duration[j];
                    let law = self.durations[j].law.fit(&counts, pc)?;
                    law.validate(m)?;
                    let current = self.durations[j].law.weighted_log_lik(&counts, m);
                    if matches!(law, Duration::Empirical { .. })
                        || law.weighted_log_lik(&counts, m) >= current
                    {
                        self.durations[j].law = law;
                    }
                }
            }
        }
        if opts.params.emission {
            let updated: Vec<Result<Emission>> = (0..k)
                .into_par_iter()
                .map(|j| {
                    let w: Vec<Vec<f64>> = posts
                        .iter()
                        .map(|p| p.gamma.column(j).iter().copied().collect())
                        .collect();
                    update_emission(&self.emissions[j], j, data, &w, pr.covariance, v)
                })
                .collect();
            self.emissions = updated.into_iter().collect::<Result<_>>()?;
        }
        Ok(())
    }

    /// EM from the current parameters; `history` holds the objective after each E-step.
    fn em(
        &mut self,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
        restart: usize,
    ) -> Result<FitReport> {
        let (_, v) = pooled_moments(data);
        let n_obs = n_observed(data).max(1) as f64;
        let mut history = Vec::new();
        let mut converged = false;
        let mut n_iter = 0;
        loop {
            let posts = self.e_step(data, clamps)?;
            let objective =
                posts.iter().map(|p| p.log_likelihood).sum::<f64>() + self.log_prior(opts, &v);
            if let Some(&prev) = history.last() {
                let prev: f64 = prev;
                if objective < prev - 1e-9 * prev.abs() {
                    return convergence(format!(
                        "EM objective decreased at iteration {n_iter}: {prev} -> {objective}"
                    ));
                }
                if (objective - prev) / n_obs < opts.tol {
                    converged = true;
                }
            }
            history.push(objective);
            if converged || n_iter >= opts.max_iter {
                break;
            }
            self.m_step(data, &posts, opts, &v)?;
            n_iter += 1;
        }
        Ok(FitReport {
            history,
            converged,
            n_iter,
            restart,
        })
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
                        invalid("ordering by mean needs continuous emissions")
                    }
                })
                .collect::<Result<_>>()?,
            Ordering::Occupancy => {
                let nu = if k > 1 {
                    gth_stationary(&self.transition)?
                } else {
                    vec![1.0]
                };
                nu.iter()
                    .zip(self.durations.iter().zip(&self.max_duration))
                    .map(|(n, (d, &m))| -(n * d.law.mean(m)))
                    .collect()
            }
        };
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| key[a].total_cmp(&key[b]).then(a.cmp(&b)));
        self.start = order.iter().map(|&i| self.start[i]).collect();
        self.transition = Mat::from_fn(k, k, |a, b| self.transition[(order[a], order[b])]);
        self.durations = order.iter().map(|&i| self.durations[i].clone()).collect();
        self.emissions = order.iter().map(|&i| self.emissions[i].clone()).collect();
        self.max_duration = order.iter().map(|&i| self.max_duration[i]).collect();
        Ok(())
    }

    /// Restart `r` initialises from `Stream::new(opts.seed, keys ++ [r])`; parts outside
    /// `init_params` come from `template`. Best final objective wins, lowest restart on ties.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit_restarts(
        template: Option<&Hsmm>,
        k: usize,
        family: &Family,
        duration: DurationFamily,
        max_duration: &[usize],
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
        keys: &[u64],
    ) -> Result<(Hsmm, FitReport)> {
        if clamps.is_some_and(|c| c.len() != data.len()) {
            return invalid("give one clamp per sequence");
        }
        if data.is_empty() {
            return invalid("fit needs at least one sequence");
        }
        if opts.ordering == Ordering::Mean && matches!(family, Family::Categorical { .. }) {
            return invalid("ordering by mean needs continuous emissions");
        }
        let ip = opts.init_params;
        let init_any = ip.start || ip.transition || ip.emission || ip.duration;
        if template.is_none() && !(ip.start && ip.transition && ip.emission && ip.duration) {
            return invalid("without a starting model every part must be initialised");
        }
        let n_init = if init_any { opts.n_init.max(1) } else { 1 };
        let results: Vec<Result<(Hsmm, FitReport)>> = (0..n_init)
            .into_par_iter()
            .map(|r| {
                let mut model = match template {
                    Some(t) if !init_any => t.clone(),
                    _ => {
                        let mut all = keys.to_vec();
                        all.push(r as u64);
                        let mut stream = Stream::new(opts.seed, &all);
                        let mut m = Hsmm::initialize(
                            k,
                            family,
                            duration,
                            data,
                            clamps,
                            &opts.priors,
                            max_duration,
                            &mut stream,
                        )?;
                        if let Some(t) = template {
                            if !ip.start {
                                m.start = t.start.clone();
                            }
                            if !ip.transition {
                                m.transition = t.transition.clone();
                            }
                            if !ip.emission {
                                m.emissions = t.emissions.clone();
                            }
                            if !ip.duration {
                                m.durations = t.durations.clone();
                            }
                            m.carry_pseudocounts(t);
                        }
                        m
                    }
                };
                let report = model.em(data, clamps, opts, r)?;
                Ok((model, report))
            })
            .collect();
        let mut best: Option<(Hsmm, FitReport)> = None;
        for res in results {
            let (m, rep) = res?;
            let better = match &best {
                None => true,
                Some((_, b)) => rep.history.last() > b.history.last(),
            };
            if better {
                best = Some((m, rep));
            }
        }
        let (mut model, report) = best.expect("at least one restart");
        model.reorder(opts.ordering)?;
        Ok((model, report))
    }

    pub fn fit(
        &mut self,
        family: &Family,
        duration: DurationFamily,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        opts: &FitOptions,
    ) -> Result<FitReport> {
        self.validate()?;
        let (model, report) = Hsmm::fit_restarts(
            Some(self),
            self.k(),
            family,
            duration,
            &self.max_duration.clone(),
            data,
            clamps,
            opts,
            &[tag::INIT],
        )?;
        *self = model;
        Ok(report)
    }
}
