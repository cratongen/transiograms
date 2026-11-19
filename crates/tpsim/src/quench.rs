use transiograms_core::rng::tag;
use transiograms_core::{Code, Grid, MISSING, Result, Stream, invalid};
use transiograms_transiogram::TransitionModel;

use crate::QuenchOptions;

/// Pair counts `F_jk(l)`, row totals `N_j(l)` and category counts over the informed active cells,
/// with the model targets of the quenching objective.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Counts {
    pub k: usize,
    pub lags: Vec<[i64; 3]>,
    pub f: Vec<f64>,
    pub rows: Vec<f64>,
    pub n: Vec<f64>,
    pub total: f64,
    target: Vec<f64>,
    proportions: Vec<f64>,
    weight: f64,
}

fn active(mask: Option<&[bool]>, cell: usize) -> bool {
    mask.is_none_or(|m| m[cell])
}

/// Informed active cell at `cell + d`, with its code.
fn neighbour(
    codes: &[Code],
    grid: &Grid,
    mask: Option<&[bool]>,
    cell: usize,
    d: [i64; 3],
) -> Option<Code> {
    let other = grid.offset(cell, d)?;
    (active(mask, other) && codes[other] != MISSING).then_some(codes[other])
}

impl Counts {
    pub fn new(
        codes: &[Code],
        grid: &Grid,
        mask: Option<&[bool]>,
        model: &TransitionModel,
        lags: &[[i64; 3]],
        weight: f64,
    ) -> Self {
        let k = model.k();
        let mut target = Vec::with_capacity(lags.len() * k * k);
        for &l in lags {
            let t = model.transition(grid.lag(l));
            for j in 0..k {
                for c in 0..k {
                    target.push(t[(j, c)]);
                }
            }
        }
        let mut out = Self {
            k,
            lags: lags.to_vec(),
            f: vec![],
            rows: vec![],
            n: vec![],
            total: 0.0,
            target,
            proportions: model.proportions.clone(),
            weight,
        };
        out.recount(codes, grid, mask);
        out
    }

    pub fn recount(&mut self, codes: &[Code], grid: &Grid, mask: Option<&[bool]>) {
        let k = self.k;
        self.f = vec![0.0; self.lags.len() * k * k];
        self.rows = vec![0.0; self.lags.len() * k];
        self.n = vec![0.0; k];
        self.total = 0.0;
        for x in 0..codes.len() {
            if !active(mask, x) || codes[x] == MISSING {
                continue;
            }
            let a = codes[x] as usize;
            self.n[a] += 1.0;
            self.total += 1.0;
            for (li, &l) in self.lags.iter().enumerate() {
                if let Some(c) = neighbour(codes, grid, mask, x, l) {
                    self.f[(li * k + a) * k + c as usize] += 1.0;
                    self.rows[li * k + a] += 1.0;
                }
            }
        }
    }

    fn row_cost(&self, li: usize, j: usize) -> f64 {
        let k = self.k;
        let n = self.rows[li * k + j];
        if n == 0.0 {
            return 0.0;
        }
        (0..k)
            .map(|c| {
                let e = self.f[(li * k + j) * k + c] / n - self.target[(li * k + j) * k + c];
                e * e
            })
            .sum()
    }

    fn proportion_cost(&self, j: usize, n_j: f64) -> f64 {
        if self.total == 0.0 {
            return 0.0;
        }
        let e = n_j / self.total - self.proportions[j];
        self.weight * e * e
    }

    pub fn objective(&self) -> f64 {
        let mut o = 0.0;
        for li in 0..self.lags.len() {
            for j in 0..self.k {
                o += self.row_cost(li, j);
            }
        }
        o + (0..self.k)
            .map(|j| self.proportion_cost(j, self.n[j]))
            .sum::<f64>()
    }

    /// Moves cell x from code a to b in the counts; `around[l] = (head code, tail code)`.
    fn shift(&mut self, around: &[(Option<Code>, Option<Code>)], a: usize, b: usize, sign: f64) {
        let k = self.k;
        for (li, &(head, tail)) in around.iter().enumerate() {
            if let Some(c) = head {
                let c = c as usize;
                self.f[(li * k + a) * k + c] -= sign;
                self.f[(li * k + b) * k + c] += sign;
                self.rows[li * k + a] -= sign;
                self.rows[li * k + b] += sign;
            }
            if let Some(e) = tail {
                let e = e as usize;
                self.f[(li * k + e) * k + a] -= sign;
                self.f[(li * k + e) * k + b] += sign;
            }
        }
        self.n[a] -= sign;
        self.n[b] += sign;
    }

    fn affected_cost(&self, around: &[(Option<Code>, Option<Code>)], a: usize, b: usize) -> f64 {
        let mut o = 0.0;
        for (li, &(_, tail)) in around.iter().enumerate() {
            o += self.row_cost(li, a) + self.row_cost(li, b);
            if let Some(e) = tail {
                let e = e as usize;
                if e != a && e != b {
                    o += self.row_cost(li, e);
                }
            }
        }
        o + self.proportion_cost(a, self.n[a]) + self.proportion_cost(b, self.n[b])
    }

    /// Exact change of the objective when cell x moves from a to b (a != b).
    pub fn delta(&mut self, around: &[(Option<Code>, Option<Code>)], a: usize, b: usize) -> f64 {
        let before = self.affected_cost(around, a, b);
        self.shift(around, a, b, 1.0);
        let after = self.affected_cost(around, a, b);
        self.shift(around, a, b, -1.0);
        after - before
    }

    pub fn around(
        &self,
        codes: &[Code],
        grid: &Grid,
        mask: Option<&[bool]>,
        x: usize,
        out: &mut Vec<(Option<Code>, Option<Code>)>,
    ) {
        out.clear();
        out.extend(self.lags.iter().map(|&l| {
            (
                neighbour(codes, grid, mask, x, l),
                neighbour(codes, grid, mask, x, [-l[0], -l[1], -l[2]]),
            )
        }));
    }

    pub fn apply(&mut self, around: &[(Option<Code>, Option<Code>)], a: usize, b: usize) {
        self.shift(around, a, b, 1.0);
    }
}

pub(crate) fn check_codes(
    codes: &[Code],
    grid: &Grid,
    mask: Option<&[bool]>,
    k: usize,
) -> Result<()> {
    if codes.len() != grid.n_cells() {
        return invalid(format!(
            "codes have {} values for {} cells",
            codes.len(),
            grid.n_cells()
        ));
    }
    crate::check_mask(grid, mask)?;
    if let Some(x) = (0..codes.len()).find(|&x| codes[x] != MISSING && codes[x] as usize >= k) {
        return invalid(format!(
            "code {} at cell {x} is not one of the model's {k} categories",
            codes[x]
        ));
    }
    Ok(())
}

/// Sequential, zero temperature; returns the objective after each sweep (index 0 = initial).
#[allow(clippy::too_many_arguments)]
pub fn quench(
    codes: &mut [Code],
    frozen: &[bool],
    grid: &Grid,
    mask: Option<&[bool]>,
    model: &TransitionModel,
    q: &QuenchOptions,
    seed: u64,
    realization: u64,
) -> Result<Vec<f64>> {
    let k = model.k();
    check_codes(codes, grid, mask, k)?;
    crate::check_quench(q)?;
    if frozen.len() != codes.len() {
        return invalid(format!(
            "frozen has {} values for {} cells",
            frozen.len(),
            codes.len()
        ));
    }
    let lags = quench_lags(model, grid, q.limit);
    let mut counts = Counts::new(codes, grid, mask, model, &lags, q.proportion_weight);
    let o0 = counts.objective();
    let mut history = vec![o0];
    if o0 == 0.0 || k < 2 {
        return Ok(history);
    }
    // A move must beat round-off in the affected-row sums, so the recounted objective never rises.
    let threshold = 1e-12 * o0;
    let cells: Vec<usize> = (0..codes.len())
        .filter(|&x| active(mask, x) && !frozen[x] && codes[x] != MISSING)
        .collect();
    let mut around = Vec::with_capacity(lags.len());
    for s in 0..q.max_iter {
        let mut order = cells.clone();
        Stream::new(seed, &[realization, tag::QUENCH, s as u64]).shuffle(&mut order);
        let mut changed = 0usize;
        for &x in &order {
            let a = codes[x] as usize;
            counts.around(codes, grid, mask, x, &mut around);
            let mut best: Option<(usize, f64)> = None;
            for b in (0..k).filter(|&b| b != a) {
                let d = counts.delta(&around, a, b);
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((b, d));
                }
            }
            if let Some((b, d)) = best
                && d < -threshold
            {
                counts.apply(&around, a, b);
                codes[x] = b as Code;
                changed += 1;
            }
        }
        #[cfg(test)]
        {
            let mut fresh = counts.clone();
            fresh.recount(codes, grid, mask);
            assert_eq!(
                fresh, counts,
                "counts differ from a recount after sweep {s}"
            );
        }
        let o = counts.objective();
        history.push(o);
        if changed == 0 || o / o0 < q.tol {
            break;
        }
    }
    Ok(history)
}

pub fn objective(
    codes: &[Code],
    grid: &Grid,
    mask: Option<&[bool]>,
    model: &TransitionModel,
    lags: &[[i64; 3]],
    proportion_weight: f64,
) -> f64 {
    Counts::new(codes, grid, mask, model, lags, proportion_weight).objective()
}

pub fn quench_lags(model: &TransitionModel, grid: &Grid, limit: f64) -> Vec<[i64; 3]> {
    let mut out = vec![];
    for dz in -1..=1i64 {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let d = [dx, dy, dz];
                let first = [dz, dy, dx].into_iter().find(|&v| v != 0);
                if first.is_none_or(|v| v < 0) {
                    continue;
                }
                if (0..3).any(|a| d[a] != 0 && grid.count[a] < 2) {
                    continue;
                }
                let unit = d.iter().filter(|&&v| v != 0).count() == 1;
                if unit || model.closeness(grid.lag(d)) >= limit {
                    out.push(d);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use transiograms_core::Mat;

    fn model() -> TransitionModel {
        let pi = Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.7, 0.0, 0.3, 0.2, 0.8, 0.0]);
        TransitionModel::from_geology(
            &[0.5, 0.3, 0.2],
            [&[4.0, 3.0, 2.0], &[3.0, 2.0, 2.0], &[2.0, 1.0, 0.8]],
            [None, None, Some(&pi)],
            2,
            [0.0; 3],
        )
        .unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn incremental_delta_equals_full_recompute(
            seed in 0u64..1000,
            cell in 0usize..60,
            b in 0usize..3,
            masked in proptest::collection::vec(0usize..60, 0..6),
        ) {
            let model = model();
            let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [5, 4, 3]).unwrap();
            let mut mask = vec![true; 60];
            for m in masked {
                mask[m] = false;
            }
            let mut s = Stream::new(seed, &[0]);
            let mut codes: Vec<Code> = (0..60)
                .map(|x| if mask[x] { s.below(3) as Code } else { MISSING })
                .collect();
            let lags = quench_lags(&model, &grid, 0.05);
            prop_assume!(mask[cell] && codes[cell] as usize != b);
            let a = codes[cell] as usize;
            let mut counts = Counts::new(&codes, &grid, Some(&mask), &model, &lags, 0.7);
            let before = counts.objective();
            let mut around = vec![];
            counts.around(&codes, &grid, Some(&mask), cell, &mut around);
            let delta = counts.delta(&around, a, b);
            counts.apply(&around, a, b);
            codes[cell] = b as Code;
            let full = objective(&codes, &grid, Some(&mask), &model, &lags, 0.7);
            prop_assert!((before + delta - full).abs() < 1e-12, "{} vs {}", before + delta, full);
            let mut fresh = counts.clone();
            fresh.recount(&codes, &grid, Some(&mask));
            prop_assert_eq!(fresh, counts);
        }
    }

    #[test]
    fn quench_lowers_the_objective_and_keeps_frozen_cells() {
        let model = model();
        let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [12, 10, 6]).unwrap();
        let n = grid.n_cells();
        let mut s = Stream::new(5, &[0]);
        let mut codes: Vec<Code> = (0..n).map(|_| s.below(3) as Code).collect();
        let frozen: Vec<bool> = (0..n).map(|x| x % 7 == 0).collect();
        let start = codes.clone();
        let q = QuenchOptions {
            max_iter: 10,
            tol: 0.0,
            ..QuenchOptions::default()
        };
        let history = quench(&mut codes, &frozen, &grid, None, &model, &q, 1, 0).unwrap();
        assert!(history.windows(2).all(|w| w[1] <= w[0]), "{history:?}");
        assert!(history.last().unwrap() < &(0.5 * history[0]), "{history:?}");
        for x in 0..n {
            if frozen[x] {
                assert_eq!(codes[x], start[x]);
            }
        }
        let lags = quench_lags(&model, &grid, q.limit);
        let full = objective(&codes, &grid, None, &model, &lags, 1.0);
        assert!((full - history.last().unwrap()).abs() < 1e-12);
    }
}
