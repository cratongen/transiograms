//! Statistical tests on transition counts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use varvite_core::special::chi2_sf;
use varvite_core::{Code, Mat, Result, SequenceSet, invalid, numerics::quasi_independence};

use crate::fit::{FitOptions, Kind, MarkovChain, embedded_fit, sequences, tally};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    G,
    Pearson,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    /// "independence", "order 2 vs 1", "symmetry", "homogeneity", "exponential".
    pub name: String,
    /// "g", "pearson", "bowker", "ks".
    pub method: String,
    pub statistic: f64,
    /// 0 for "ks".
    pub df: usize,
    pub p_value: f64,
    /// Null-hypothesis expected counts when tabular.
    #[serde(with = "varvite_core::serde_mat::option")]
    pub expected: Option<Mat>,
    /// Fraction of allowed cells with expected < 5.
    pub small_expected: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderTable {
    pub orders: Vec<usize>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub log_likelihood: Vec<f64>,
    pub n_params: Vec<usize>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub aic: Vec<f64>,
    #[serde(with = "varvite_core::serde_mat::values")]
    pub bic: Vec<f64>,
    pub n_transitions: usize,
}

fn method_name(method: Method) -> &'static str {
    match method {
        Method::G => "g",
        Method::Pearson => "pearson",
    }
}

/// `(statistic, cells with E < 5, allowed cells)` over `(observed, expected, allowed)`.
fn statistic(cells: impl Iterator<Item = (f64, f64, bool)>, method: Method) -> (f64, usize, usize) {
    let (mut stat, mut small, mut allowed) = (0.0, 0, 0);
    for (o, e, ok) in cells {
        if !ok {
            continue;
        }
        allowed += 1;
        if e < 5.0 {
            small += 1;
        }
        if e <= 0.0 {
            continue;
        }
        stat += match method {
            Method::Pearson => (o - e) * (o - e) / e,
            Method::G if o > 0.0 => 2.0 * o * (o / e).ln(),
            Method::G => 0.0,
        };
    }
    (stat.max(0.0), small, allowed)
}

fn result(
    name: &str,
    method: &str,
    stat: f64,
    df: usize,
    expected: Option<Mat>,
    small: usize,
    allowed: usize,
) -> TestResult {
    TestResult {
        name: name.into(),
        method: method.into(),
        statistic: stat,
        df,
        p_value: if df == 0 {
            1.0
        } else {
            chi2_sf(stat, df as f64)
        },
        expected,
        small_expected: if allowed == 0 {
            0.0
        } else {
            small as f64 / allowed as f64
        },
    }
}

/// Categories kept after repeatedly dropping those with a zero row or column margin.
fn kept(n: &Mat) -> Vec<usize> {
    let mut keep: Vec<usize> = (0..n.nrows()).collect();
    loop {
        let next: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&i| {
                keep.iter().map(|&j| n[(i, j)]).sum::<f64>() > 0.0
                    && keep.iter().map(|&j| n[(j, i)]).sum::<f64>() > 0.0
            })
            .collect();
        if next.len() == keep.len() {
            return keep;
        }
        keep = next;
    }
}

/// Expected counts under (quasi-)independence, full size with zeros elsewhere, and the kept rows
/// and columns. Quasi-independence keeps the categories left by `kept`; the ordinary contingency
/// keeps every row and every column with a positive margin.
fn independence_expected(
    n: &Mat,
    structural_diagonal: bool,
) -> Result<(Mat, Vec<usize>, Vec<usize>)> {
    let m = n.nrows();
    let mut e = Mat::zeros(m, m);
    if structural_diagonal {
        let keep = kept(n);
        let r = keep.len();
        let rows: Vec<f64> = keep
            .iter()
            .map(|&i| keep.iter().map(|&j| n[(i, j)]).sum())
            .collect();
        let cols: Vec<f64> = keep
            .iter()
            .map(|&j| keep.iter().map(|&i| n[(i, j)]).sum())
            .collect();
        if r <= 2 {
            return invalid(format!(
                "an embedded chain with {r} categories always alternates; no test exists"
            ));
        }
        let q = quasi_independence(&rows, &cols, 1e-13, 100_000)?;
        for (a, &i) in keep.iter().enumerate() {
            for (b, &j) in keep.iter().enumerate() {
                e[(i, j)] = q[(a, b)];
            }
        }
        return Ok((e, keep.clone(), keep));
    }
    let rows: Vec<f64> = (0..m).map(|i| n.row(i).sum()).collect();
    let cols: Vec<f64> = (0..m).map(|j| n.column(j).sum()).collect();
    let rk: Vec<usize> = (0..m).filter(|&i| rows[i] > 0.0).collect();
    let ck: Vec<usize> = (0..m).filter(|&j| cols[j] > 0.0).collect();
    if rk.len() < 2 || ck.len() < 2 {
        return invalid(format!(
            "the independence test needs at least 2 categories with exits and 2 with entries, got {} and {}",
            rk.len(),
            ck.len()
        ));
    }
    let total: f64 = rows.iter().sum();
    for &i in &rk {
        for &j in &ck {
            e[(i, j)] = rows[i] * cols[j] / total;
        }
    }
    Ok((e, rk, ck))
}

impl MarkovChain {
    fn counts_or_raise(&self) -> Result<&Mat> {
        match &self.counts {
            Some(c) => Ok(c),
            None => invalid("this test needs transition counts: fit the chain from data"),
        }
    }

    fn structural_diagonal(&self) -> bool {
        self.kind == Kind::Embedded && !self.self_transitions
    }

    /// Order 1 against independence: quasi-independence for embedded chains, ordinary
    /// contingency otherwise. On a fixed chain bed persistence makes this test reject almost
    /// always; it says nothing about juxtaposition.
    pub fn independence_test(&self, method: Method) -> Result<TestResult> {
        let n = self.counts_or_raise()?;
        let structural = self.structural_diagonal();
        let (e, rk, ck) = independence_expected(n, structural)?;
        let r = rk.len();
        let df = if structural {
            r * r - 3 * r + 1
        } else {
            (r - 1) * (ck.len() - 1)
        };
        let cells = rk.iter().flat_map(|&i| ck.iter().map(move |&j| (i, j)));
        let (stat, small, allowed) = statistic(
            cells.map(|(i, j)| (n[(i, j)], e[(i, j)], !(structural && i == j))),
            method,
        );
        Ok(result(
            "independence",
            method_name(method),
            stat,
            df,
            Some(e),
            small,
            allowed,
        ))
    }

    /// Bowker on counts.
    pub fn symmetry_test(&self) -> Result<TestResult> {
        let n = self.counts_or_raise()?;
        let m = n.nrows();
        let mut e = n.clone();
        let (mut stat, mut df, mut small) = (0.0, 0, 0);
        for i in 0..m {
            for j in (i + 1)..m {
                let s = n[(i, j)] + n[(j, i)];
                e[(i, j)] = s / 2.0;
                e[(j, i)] = s / 2.0;
                if s > 0.0 {
                    stat += (n[(i, j)] - n[(j, i)]).powi(2) / s;
                    df += 1;
                    if s / 2.0 < 5.0 {
                        small += 2;
                    }
                }
            }
        }
        Ok(result(
            "symmetry",
            "bowker",
            stat,
            df,
            Some(e),
            small,
            2 * df,
        ))
    }

    /// `Pi_obs - Pi_expected` under quasi-independence (Miall 1973), or under ordinary
    /// independence when the diagonal is not structural; positive entries are preferred paths.
    /// Rows and columns of categories dropped for a zero margin are NaN.
    pub fn difference_matrix(&self) -> Result<Mat> {
        let n = self.counts_or_raise()?;
        let (e, rk, ck) = independence_expected(n, self.structural_diagonal())?;
        let m = n.nrows();
        let p = self.transition.matrix();
        let mut d = Mat::from_element(m, m, f64::NAN);
        for &i in &rk {
            let s: f64 = e.row(i).sum();
            for &j in &ck {
                d[(i, j)] = p[(i, j)] - e[(i, j)] / s;
            }
        }
        Ok(d)
    }
}

/// Tallies of `(history, next)` over every position `t >= skip` of every sequence, the history
/// being the `order` codes before `t`.
fn history_counts(
    seqs: &[Vec<Code>],
    m: usize,
    order: usize,
    skip: usize,
) -> BTreeMap<Vec<Code>, Vec<f64>> {
    let mut out: BTreeMap<Vec<Code>, Vec<f64>> = BTreeMap::new();
    for s in seqs {
        for t in skip.max(order)..s.len() {
            let row = out
                .entry(s[t - order..t].to_vec())
                .or_insert_with(|| vec![0.0; m]);
            row[s[t] as usize] += 1.0;
        }
    }
    out
}

/// Raises unless the `history_counts` tables of `order` (`copies` of them) plus `extra` bytes fit
/// the memory budget.
fn check_histories(
    seqs: &[Vec<Code>],
    m: usize,
    order: usize,
    copies: u64,
    extra: u64,
) -> Result<()> {
    let positions: u64 = seqs
        .iter()
        .map(|s| s.len().saturating_sub(order) as u64)
        .sum();
    // Per distinct history (at most one per position): its key, its row, two Vec headers and a
    // B-tree slot.
    let per = (order as u64)
        .checked_mul(4)
        .and_then(|b| b.checked_add(m as u64 * 8 + 80));
    match per
        .and_then(|p| p.checked_mul(positions))
        .and_then(|b| b.checked_mul(copies)?.checked_add(extra))
    {
        Some(b) => varvite_core::memory::check(b, None, "order test history tables"),
        None => invalid(format!(
            "order {order} with {m} categories overflows the address space"
        )),
    }
}

fn fixed_order_test(
    seqs: &[Vec<Code>],
    m: usize,
    order: usize,
    method: Method,
) -> Result<TestResult> {
    let n_hist = u32::try_from(order)
        .ok()
        .and_then(|o| m.checked_pow(o))
        .filter(|&h| h.checked_mul(m * 8).is_some());
    let Some(n_hist) = n_hist else {
        return invalid(format!(
            "order {order} with {m} categories gives more than 2^64 histories; lower the order"
        ));
    };
    let transitions: usize = seqs.iter().map(|s| s.len().saturating_sub(order)).sum();
    if n_hist > transitions {
        return invalid(format!(
            "order {order} with {m} categories has {n_hist} histories but only {transitions} \
             transitions; lower the order"
        ));
    }
    check_histories(seqs, m, order, 2, (n_hist * m * 8) as u64)?;
    let fine = history_counts(seqs, m, order, order);
    let mut coarse: BTreeMap<Vec<Code>, Vec<f64>> = BTreeMap::new();
    for (s, row) in &fine {
        let c = coarse
            .entry(s[1..].to_vec())
            .or_insert_with(|| vec![0.0; m]);
        for j in 0..m {
            c[j] += row[j];
        }
    }
    if fine.is_empty() {
        return invalid(format!(
            "no run is longer than {order} steps; give longer sequences"
        ));
    }
    let mut expected = Mat::zeros(n_hist, m);
    let index = |s: &[Code]| s.iter().fold(0usize, |a, &c| a * m + c as usize);
    let mut cells = Vec::new();
    for (s, row) in &fine {
        let c = &coarse[&s[1..]];
        let ct: f64 = c.iter().sum();
        let rt: f64 = row.iter().sum();
        for j in 0..m {
            let e = rt * c[j] / ct;
            expected[(index(s), j)] = e;
            cells.push((row[j], e, true));
        }
    }
    let mut df = 0;
    for (s1, c) in &coarse {
        let rows = fine.keys().filter(|s| &s[1..] == s1.as_slice()).count();
        let cols = c.iter().filter(|&&v| v > 0.0).count();
        df += rows.saturating_sub(1) * cols.saturating_sub(1);
    }
    let (stat, small, allowed) = statistic(cells.into_iter(), method);
    let name = if order == 1 {
        "independence".to_string()
    } else {
        format!("order {order} vs {}", order - 1)
    };
    Ok(result(
        &name,
        method_name(method),
        stat,
        df,
        Some(expected),
        small,
        allowed,
    ))
}

fn embedded_order2_test(seqs: &[Vec<Code>], m: usize, method: Method) -> Result<TestResult> {
    let mut n = vec![0.0; m * m * m];
    let at = |i: usize, j: usize, k: usize| (i * m + j) * m + k;
    for s in seqs {
        for w in s.windows(3) {
            n[at(w[0] as usize, w[1] as usize, w[2] as usize)] += 1.0;
        }
    }
    let mut expected = Mat::zeros(m * m, m);
    let mut cells = Vec::new();
    let mut df = 0;
    for j in 0..m {
        let nij: Vec<f64> = (0..m)
            .map(|i| (0..m).map(|k| n[at(i, j, k)]).sum())
            .collect();
        let njk: Vec<f64> = (0..m)
            .map(|k| (0..m).map(|i| n[at(i, j, k)]).sum())
            .collect();
        let nj: f64 = nij.iter().sum();
        if nj == 0.0 {
            continue;
        }
        let rows = (0..m).filter(|&i| i != j && nij[i] > 0.0).count();
        let cols = (0..m).filter(|&k| k != j && njk[k] > 0.0).count();
        df += rows.saturating_sub(1) * cols.saturating_sub(1);
        for i in (0..m).filter(|&i| i != j) {
            for k in (0..m).filter(|&k| k != j) {
                let e = nij[i] * njk[k] / nj;
                expected[(i * m + j, k)] = e;
                cells.push((n[at(i, j, k)], e, nij[i] > 0.0));
            }
        }
    }
    if df == 0 {
        return invalid(
            "no middle category has two predecessors and two successors; no order 2 test exists",
        );
    }
    let (stat, small, allowed) = statistic(cells.into_iter(), method);
    Ok(result(
        "order 2 vs 1",
        method_name(method),
        stat,
        df,
        Some(expected),
        small,
        allowed,
    ))
}

/// Order `order` against `order - 1`. Fixed: Anderson and Goodman on k-histories, the first
/// `order` states of every run dropped. Embedded (no self-transitions): order 1 is the
/// quasi-independence test, order 2 the triplet test; higher orders raise.
pub fn order_test(
    seqs: &SequenceSet,
    opts: &FitOptions,
    order: usize,
    method: Method,
) -> Result<TestResult> {
    if order == 0 {
        return invalid("order must be at least 1");
    }
    let m = seqs.n_categories;
    match opts.kind {
        Kind::Fixed { .. } => fixed_order_test(&sequences(seqs, opts)?, m, order, method),
        Kind::Embedded => {
            if order == 1 {
                return MarkovChain::fit(seqs, opts)?.independence_test(method);
            }
            if order > 2 {
                return invalid(format!(
                    "embedded order tests go up to order 2, got {order}"
                ));
            }
            if opts.self_transitions {
                return invalid("the embedded order 2 test needs self_transitions = false");
            }
            embedded_order2_test(&sequences(seqs, opts)?, m, method)
        }
        Kind::Continuous => invalid("order tests take kind Embedded or Fixed"),
    }
}

/// Fixed only. Every order is fitted on the same transitions: the first `max_order` states of
/// every run are dropped.
pub fn select_order(seqs: &SequenceSet, opts: &FitOptions, max_order: usize) -> Result<OrderTable> {
    if !matches!(opts.kind, Kind::Fixed { .. }) {
        return invalid("select_order needs kind Fixed");
    }
    let m = seqs.n_categories;
    let n_params = |k: usize| {
        u32::try_from(k)
            .ok()
            .and_then(|k| m.checked_pow(k)?.checked_mul(m.saturating_sub(1)))
    };
    if n_params(max_order).is_none() {
        return invalid(format!(
            "max_order {max_order} with {m} categories gives more than 2^64 parameters; lower it"
        ));
    }
    let s = sequences(seqs, opts)?;
    check_histories(&s, m, max_order, 1, 0)?;
    let n: usize = s.iter().map(|x| x.len().saturating_sub(max_order)).sum();
    if n == 0 {
        return invalid(format!(
            "no run is longer than {max_order} steps; give longer sequences"
        ));
    }
    let mut t = OrderTable {
        orders: vec![],
        log_likelihood: vec![],
        n_params: vec![],
        aic: vec![],
        bic: vec![],
        n_transitions: n,
    };
    for k in 0..=max_order {
        let ll: f64 = history_counts(&s, m, k, max_order)
            .values()
            .map(|row| {
                let tot: f64 = row.iter().sum();
                row.iter()
                    .filter(|&&c| c > 0.0)
                    .map(|&c| c * (c / tot).ln())
                    .sum::<f64>()
            })
            .sum();
        let np = n_params(k).expect("bounded by n_params(max_order)");
        t.orders.push(k);
        t.log_likelihood.push(ll);
        t.n_params.push(np);
        t.aic.push(-2.0 * ll + 2.0 * np as f64);
        t.bic.push(-2.0 * ll + np as f64 * (n as f64).ln());
    }
    Ok(t)
}

/// Pooled against per-group transition matrices. `expected` stacks the groups' tables
/// (group-major, `H m` rows).
pub fn homogeneity_test(
    groups: &[SequenceSet],
    opts: &FitOptions,
    method: Method,
) -> Result<TestResult> {
    if groups.len() < 2 {
        return invalid(format!("give at least 2 groups, got {}", groups.len()));
    }
    let m = groups[0].n_categories;
    if let Some(h) = groups.iter().position(|g| g.n_categories != m) {
        return invalid(format!(
            "group {h} has {} categories but group 0 has {m}",
            groups[h].n_categories
        ));
    }
    let counts: Vec<Mat> = groups
        .iter()
        .map(|g| Ok(tally(&sequences(g, opts)?, m)))
        .collect::<Result<_>>()?;
    let pooled = counts.iter().fold(Mat::zeros(m, m), |a, c| a + c);
    let structural = embedded_fit(opts);
    let mut expected = Mat::zeros(groups.len() * m, m);
    let mut cells = Vec::new();
    let mut df = 0;
    for i in 0..m {
        let pr: f64 = pooled.row(i).sum();
        if pr == 0.0 {
            continue;
        }
        let groups_i = counts.iter().filter(|c| c.row(i).sum() > 0.0).count();
        let cols_i = (0..m)
            .filter(|&j| !(structural && i == j) && pooled[(i, j)] > 0.0)
            .count();
        df += groups_i.saturating_sub(1) * cols_i.saturating_sub(1);
        for (h, c) in counts.iter().enumerate() {
            let hr: f64 = c.row(i).sum();
            for j in (0..m).filter(|&j| !(structural && i == j)) {
                let e = hr * pooled[(i, j)] / pr;
                expected[(h * m + i, j)] = e;
                cells.push((c[(i, j)], e, hr > 0.0));
            }
        }
    }
    let (stat, small, allowed) = statistic(cells.into_iter(), method);
    Ok(result(
        "homogeneity",
        method_name(method),
        stat,
        df,
        Some(expected),
        small,
        allowed,
    ))
}
