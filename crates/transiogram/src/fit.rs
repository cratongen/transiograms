use transiograms_core::numerics::{golden_section, nelder_mead};
use transiograms_core::{Mat, RateMatrix, Result, TransitionMatrix, convergence, invalid};

use crate::experimental::Experimental;
use crate::model1d::{Entry, Model1D, Structure, check_proportions};

fn check_experimental(exp: &Experimental) -> Result<usize> {
    let n = exp.lags.len();
    if n == 0 {
        return invalid("the experimental transiogram has no lags");
    }
    if exp.mean_lags.len() != n
        || exp.probabilities.len() != n
        || exp.tail_totals.len() != n
        || exp.counts.len() != n
    {
        return invalid("the experimental transiogram arrays have different lengths");
    }
    let k = exp.probabilities[0].nrows();
    if k == 0
        || exp
            .probabilities
            .iter()
            .any(|p| p.nrows() != k || p.ncols() != k)
        || exp.tail_totals.iter().any(|t| t.len() != k)
    {
        return invalid("the experimental transiogram matrices must all be k x k");
    }
    Ok(k)
}

/// Mean lag when known, else the nominal lag.
fn effective_lags(exp: &Experimental) -> Vec<f64> {
    exp.lags
        .iter()
        .zip(&exp.mean_lags)
        .map(|(&l, &m)| if m.is_finite() && m > 0.0 { m } else { l })
        .collect()
}

/// `(h, i, N_i, row of t_hat)` for every usable row.
fn usable_rows(exp: &Experimental) -> Vec<(f64, usize, f64, Vec<f64>)> {
    let lags = effective_lags(exp);
    let mut out = Vec::new();
    for (l, &h) in lags.iter().enumerate() {
        if !(h > 0.0 && h.is_finite()) {
            continue;
        }
        for (i, &n) in exp.tail_totals[l].iter().enumerate() {
            let row: Vec<f64> = exp.probabilities[l].row(i).iter().copied().collect();
            if n > 0.0 && row.iter().all(|v| v.is_finite()) {
                out.push((h, i, n, row));
            }
        }
    }
    out
}

fn initial_rates(exp: &Experimental, k: usize) -> Result<RateMatrix> {
    let lags = effective_lags(exp);
    let Some(l) = (0..lags.len()).find(|&l| {
        lags[l] > 0.0
            && exp.tail_totals[l].iter().all(|&n| n > 0.0)
            && exp.probabilities[l].iter().all(|v| v.is_finite())
    }) else {
        return invalid("no lag has every category as a tail: pass start=");
    };
    let t = &exp.probabilities[l];
    if let Ok((r, _)) =
        TransitionMatrix::new(t.clone()).and_then(|m| RateMatrix::from_transition(&m, lags[l]))
    {
        return Ok(r);
    }
    let total: f64 = exp.tail_totals[l].iter().sum();
    let p: Vec<f64> = exp.tail_totals[l].iter().map(|n| n / total).collect();
    let lengths: Vec<f64> = (0..k)
        .map(|i| lags[l] / (1.0 - t[(i, i)]).max(1e-6))
        .collect();
    RateMatrix::max_entropy(&p, &lengths).or_else(|_| {
        convergence("no starting rates from the first lag (no matrix logarithm, no maximum-entropy chain): pass start=")
    })
}

impl Model1D {
    pub fn fit_markov(
        exp: &Experimental,
        proportions: Option<&[f64]>,
        background: usize,
        start: Option<&RateMatrix>,
    ) -> Result<Self> {
        let k = check_experimental(exp)?;
        if background >= k {
            return invalid(format!("background {background} is not a category of {k}"));
        }
        let start = match start {
            Some(r) if r.k() != k => {
                return invalid(format!("start rates are {}x{}, not {k}x{k}", r.k(), r.k()));
            }
            Some(r) => r.clone(),
            None => initial_rates(exp, k)?,
        };
        let p = match proportions {
            Some(p) => p.to_vec(),
            None => start.stationary()?,
        };
        check_proportions(&p, k)?;
        if k == 1 {
            return Ok(Model1D::Markov {
                proportions: p,
                rates: RateMatrix::new(Mat::zeros(1, 1))?,
            });
        }
        let q = background;
        let others: Vec<usize> = (0..k).filter(|&i| i != q).collect();
        let build = |x: &[f64]| -> Result<RateMatrix> {
            let mut partial = Mat::zeros(k, k);
            let mut it = x.iter();
            for &i in &others {
                for &j in &others {
                    let v = it.next().expect("(k-1)^2 parameters").exp();
                    partial[(i, j)] = if i == j { -v } else { v };
                }
            }
            RateMatrix::from_partial(&partial, &p, q)
        };
        let s = start.matrix();
        let floor = 1e-6 * s.iter().fold(0.0, |a: f64, v| a.max(v.abs()));
        let mut shrink = 1.0;
        let x0 = loop {
            let x: Vec<f64> = others
                .iter()
                .flat_map(|&i| others.iter().map(move |&j| (i, j)))
                .map(|(i, j)| {
                    if i == j {
                        (-s[(i, i)]).ln()
                    } else {
                        (shrink * s[(i, j)]).max(floor).ln()
                    }
                })
                .collect();
            if build(&x).is_ok() {
                break x;
            }
            shrink *= 0.5;
            if shrink < 1e-12 {
                return convergence("no valid starting rates for these proportions");
            }
        };
        let rows = usable_rows(exp);
        if rows.is_empty() {
            return invalid("the experimental transiogram has no non-empty rows");
        }
        let mut lags: Vec<f64> = rows.iter().map(|r| r.0).collect();
        lags.sort_by(f64::total_cmp);
        lags.dedup();
        let objective = |x: &[f64]| -> f64 {
            let Ok(r) = build(x) else {
                return f64::INFINITY;
            };
            let ts: Vec<Mat> = lags
                .iter()
                .map(|&h| r.transition(h).matrix().clone())
                .collect();
            rows.iter()
                .map(|(h, i, n, row)| {
                    let t = &ts[lags.partition_point(|v| v < h)];
                    n * row
                        .iter()
                        .enumerate()
                        .map(|(j, v)| (v - t[(*i, j)]).powi(2))
                        .sum::<f64>()
                })
                .sum()
        };
        let m = (k - 1) * (k - 1);
        let (x, _) = nelder_mead(objective, &x0, 0.2, 1e-10, 2000 * m);
        let rates = build(&x)?;
        Ok(Model1D::Markov {
            proportions: p,
            rates,
        })
    }

    /// `structures`: one for every entry, or k * k (row-major; the complement entries are ignored).
    pub fn fit_entrywise(
        exp: &Experimental,
        proportions: &[f64],
        structures: &[Structure],
        complement: &[usize],
    ) -> Result<Self> {
        let k = check_experimental(exp)?;
        check_proportions(proportions, k)?;
        if structures.len() != 1 && structures.len() != k * k {
            return invalid(format!(
                "give 1 or k * k = {} structures, not {}",
                k * k,
                structures.len()
            ));
        }
        if complement.len() != k || complement.iter().any(|&c| c >= k) {
            return invalid(format!("give {k} complement columns, each below {k}"));
        }
        let rows = usable_rows(exp);
        let positive: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let (Some(first), Some(last)) = (
            positive.iter().copied().reduce(f64::min),
            positive.iter().copied().reduce(f64::max),
        ) else {
            return invalid("the experimental transiogram has no non-empty rows");
        };
        let (lo, hi) = ((0.01 * first).ln(), (100.0 * last).ln());
        let mut entries = vec![None; k * k];
        for i in 0..k {
            for j in (0..k).filter(|&j| j != complement[i]) {
                let structure = structures[if structures.len() == 1 { 0 } else { i * k + j }];
                let entry = |ln_range: f64| Entry {
                    structure,
                    range: ln_range.exp(),
                    nugget: 0.0,
                };
                let loss = |ln_range: f64| {
                    let e = entry(ln_range);
                    rows.iter()
                        .filter(|r| r.1 == i)
                        .map(|(h, _, n, row)| n * (row[j] - e.value(i, j, proportions, *h)).powi(2))
                        .sum::<f64>()
                };
                let (x, _) = golden_section(loss, lo, hi, 1e-10);
                entries[i * k + j] = Some(entry(x));
            }
        }
        Self::entrywise(proportions.to_vec(), entries, complement.to_vec(), last)
    }
}

impl Model1D {
    /// Experimental minus model `t_ij` at each lag of `exp` (the mean lag of its pairs when known,
    /// the nominal lag otherwise): the misfit `fit_markov` and `fit_entrywise` minimise, weighted
    /// there by the tail totals. NaN where the experimental entry is NaN.
    pub fn residuals(&self, exp: &Experimental) -> Result<Vec<Mat>> {
        let k = check_experimental(exp)?;
        if self.proportions().len() != k {
            return invalid(format!(
                "the model has {} categories, the experimental transiogram {k}",
                self.proportions().len()
            ));
        }
        Ok(effective_lags(exp)
            .iter()
            .zip(&exp.probabilities)
            .map(|(&h, t)| t - self.transition(h))
            .collect())
    }
}
