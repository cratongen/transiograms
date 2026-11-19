use std::borrow::Borrow;

use rayon::prelude::*;
use transiograms_core::{Code, Grid, MISSING, Mat, Prediction, Result, Snapped, invalid, memory};
use transiograms_transiogram::TransitionModel;

use crate::Options;
use crate::search::Neighbourhood;

/// Block weights `W` (nK x K) of `sum_beta T(x_beta - x_alpha) W_beta = T(x_0 - x_alpha)`.
/// `between(a, b)` gives `T(x_b - x_a)`, `to_target(a)` gives `T(x_0 - x_a)`; `p` are the
/// model proportions.
///
/// Because `T(h) 1 = 1` and `p^T T(h) = p^T`, the full system always has an (n - 1)-dimensional
/// null space `(1 b_1, ..., 1 b_n)`, `sum b = 0`, and as many redundant equations. Probabilities
/// do not depend on the null-space component, so row `q = argmax p` of every `W_alpha` except
/// `W_free` is fixed at 0 and the matching equations are dropped. The reduced system is
/// nonsingular for distinct data and keeps `sum_k Pr_k = 1` to round-off. With `free` the datum
/// closest to the target, a datum the Markov property screens out gets exactly `W_alpha = 0`.
pub(crate) fn weights<A, B>(
    p: &[f64],
    n: usize,
    free: usize,
    between: impl Fn(usize, usize) -> A,
    to_target: impl Fn(usize) -> B,
    svd_cutoff: f64,
) -> Result<Mat>
where
    A: Borrow<Mat>,
    B: Borrow<Mat>,
{
    let k = p.len();
    let q = (0..k).fold(0, |m, j| if p[j] > p[m] { j } else { m });
    let keep: Vec<usize> = (0..n * k)
        .filter(|&i| i / k == free || i % k != q)
        .collect();
    let mut position = vec![usize::MAX; n * k];
    for (r, &i) in keep.iter().enumerate() {
        position[i] = r;
    }
    let m = keep.len();
    let mut a = Mat::zeros(m, m);
    let mut b = Mat::zeros(m, k);
    for alpha in 0..n {
        for beta in 0..n {
            let t = between(alpha, beta);
            let t = t.borrow();
            for i in 0..k {
                let r = position[alpha * k + i];
                if r == usize::MAX {
                    continue;
                }
                for j in 0..k {
                    let c = position[beta * k + j];
                    if c != usize::MAX {
                        a[(r, c)] = t[(i, j)];
                    }
                }
            }
        }
        let t = to_target(alpha);
        let t = t.borrow();
        for i in 0..k {
            let r = position[alpha * k + i];
            if r != usize::MAX {
                for j in 0..k {
                    b[(r, j)] = t[(i, j)];
                }
            }
        }
    }
    let svd = a.clone().svd(true, true);
    let sigma_max = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let eps = svd_cutoff * sigma_max;
    let failed = || transiograms_core::convergence("cokriging system could not be solved by SVD");
    let mut reduced = svd.solve(&b, eps).or_else(|_| failed())?;
    // nalgebra's SVD left residuals of 1.6e-8 on a well-conditioned system; refinement fixes it.
    for _ in 0..2 {
        let residual = &b - &a * &reduced;
        reduced += svd.solve(&residual, eps).or_else(|_| failed())?;
    }
    if !reduced.iter().all(|v| v.is_finite()) {
        return failed();
    }
    let mut w = Mat::zeros(n * k, k);
    for (r, &i) in keep.iter().enumerate() {
        w.row_mut(i).copy_from(&reduced.row(r));
    }
    Ok(w)
}

/// `Pr_k = sum_alpha W_alpha[c_alpha, k]`.
pub(crate) fn combine(w: &Mat, codes: impl Iterator<Item = Code>, k: usize) -> Vec<f64> {
    let mut pr = vec![0.0; k];
    for (alpha, c) in codes.enumerate() {
        for j in 0..k {
            pr[j] += w[(alpha * k + c as usize, j)];
        }
    }
    pr
}

/// Index of the first datum at the smallest Euclidean distance from `target`.
fn nearest(target: [f64; 3], xs: impl Iterator<Item = [f64; 3]>) -> usize {
    let dist2 = |x: [f64; 3]| (0..3).map(|a| (target[a] - x[a]).powi(2)).sum::<f64>();
    xs.enumerate()
        .fold((0, f64::INFINITY), |(m, dm), (a, x)| {
            let d = dist2(x);
            if d < dm { (a, d) } else { (m, dm) }
        })
        .0
}

/// Probabilities before order-relation correction, summing to 1.
pub fn cokrige(
    model: &TransitionModel,
    data: &[([f64; 3], Code)],
    target: [f64; 3],
    svd_cutoff: f64,
) -> Result<Vec<f64>> {
    let k = model.k();
    if !(svd_cutoff.is_finite() && svd_cutoff >= 0.0) {
        return invalid(format!(
            "svd_cutoff must be finite and non-negative, got {svd_cutoff}"
        ));
    }
    if let Some((i, (_, c))) = data.iter().enumerate().find(|(_, d)| d.1 as usize >= k) {
        return invalid(format!(
            "datum {i} has code {c}, not one of the model's {k} categories"
        ));
    }
    if !target.iter().all(|v| v.is_finite()) {
        return invalid(format!("target {target:?} has a non-finite coordinate"));
    }
    if let Some((i, (x, _))) = data
        .iter()
        .enumerate()
        .find(|(_, d)| !d.0.iter().all(|v| v.is_finite()))
    {
        return invalid(format!("datum {i} at {x:?} has a non-finite coordinate"));
    }
    if data.is_empty() {
        return Ok(model.proportions.clone());
    }
    memory::check(
        crate::system_bytes(data.len(), k),
        None,
        "tpsim cokriging system",
    )?;
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let w = weights(
        &model.proportions,
        data.len(),
        nearest(target, data.iter().map(|d| d.0)),
        |a, b| model.transition(sub(data[b].0, data[a].0)),
        |a| model.transition(sub(target, data[a].0)),
        svd_cutoff,
    )?;
    Ok(combine(&w, data.iter().map(|d| d.1), k))
}

/// Cokriging from grid neighbours given as offsets from the target cell.
pub(crate) fn cokrige_offsets(
    nb: &Neighbourhood,
    data: &[([i64; 3], Code)],
    svd_cutoff: f64,
) -> Result<Vec<f64>> {
    let k = nb.model.k();
    if data.is_empty() {
        return Ok(nb.model.proportions.clone());
    }
    let sub = |a: [i64; 3], b: [i64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let w = weights(
        &nb.model.proportions,
        data.len(),
        0,
        |a, b| nb.transition(sub(data[b].0, data[a].0)),
        |a| nb.transition(sub([0; 3], data[a].0)),
        svd_cutoff,
    )?;
    Ok(combine(&w, data.iter().map(|d| d.1), k))
}

/// Clip to [0, 1] and renormalise; true if changed.
pub fn correct(probabilities: &mut [f64]) -> bool {
    let changed = probabilities.iter().any(|&p| !(0.0..=1.0).contains(&p));
    if !changed {
        return false;
    }
    for p in probabilities.iter_mut() {
        *p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
    }
    let total: f64 = probabilities.iter().sum();
    let n = probabilities.len() as f64;
    for p in probabilities.iter_mut() {
        *p = if total > 0.0 { *p / total } else { 1.0 / n };
    }
    true
}

pub fn predict(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Prediction> {
    let k = crate::check_inputs(model, grid, mask, hard, opts)?;
    let n_cells = grid.n_cells();
    let budget = opts.memory.or_else(memory::available_bytes);
    let per_cell = (k * size_of::<f64>() + 2 * size_of::<Code>() + size_of::<f64>()) as u64;
    let cells = (n_cells as u64).saturating_mul(per_cell);
    memory::check(cells, budget, "tp_cokriging probabilities")?;
    let (table, max_data) = crate::search_table(model, grid, opts, budget, cells)?;
    let nb = Neighbourhood::new(model, grid, &table);
    let mut informed = vec![MISSING; n_cells];
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        informed[cell] = code;
    }
    let Some(total) = n_cells.checked_mul(k) else {
        return invalid(format!(
            "{n_cells} cells of {k} probabilities overflow the address space"
        ));
    };
    let mut probabilities = vec![f64::NAN; total];
    let n_corrected = probabilities
        .par_chunks_mut(k)
        .enumerate()
        .map(|(cell, row)| {
            if mask.is_some_and(|m| !m[cell]) {
                return Ok(0);
            }
            if informed[cell] != MISSING {
                row.fill(0.0);
                row[informed[cell] as usize] = 1.0;
                return Ok(0);
            }
            let mut data = Vec::with_capacity(max_data);
            nb.search(cell, &informed, max_data, opts.max_per_octant, &mut data);
            let mut pr = cokrige_offsets(&nb, &data, opts.svd_cutoff)?;
            let changed = correct(&mut pr);
            row.copy_from_slice(&pr);
            Ok(usize::from(changed))
        })
        .try_reduce(|| 0, |a, b| Ok(a + b))?;
    let estimated = (0..n_cells)
        .filter(|&c| informed[c] == MISSING && !mask.is_some_and(|m| !m[c]))
        .count();
    drop(informed);
    let corrected = if estimated == 0 {
        0.0
    } else {
        n_corrected as f64 / estimated as f64
    };
    Ok(Prediction::from_probabilities(k, probabilities, corrected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_clips_and_renormalises() {
        let mut ok = vec![0.2, 0.3, 0.5];
        assert!(!correct(&mut ok));
        assert_eq!(ok, vec![0.2, 0.3, 0.5]);
        let mut bad = vec![-0.1, 0.6, 0.5];
        assert!(correct(&mut bad));
        assert!((bad[0]).abs() == 0.0 && (bad[1] - 0.6 / 1.1).abs() < 1e-15);
        assert!((bad.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let mut over = vec![1.2, -0.2];
        assert!(correct(&mut over));
        assert_eq!(over, vec![1.0, 0.0]);
    }

    /// x1 < x2 < x0 on a line: Chapman-Kolmogorov screening gives W1 = 0 and W2 = T(x0 - x2).
    #[test]
    fn screening_on_a_line() {
        let pi = Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.7, 0.0, 0.3, 0.2, 0.8, 0.0]);
        let model = TransitionModel::from_geology(
            &[0.5, 0.3, 0.2],
            [&[6.0, 4.0, 3.0], &[5.0, 3.0, 2.0], &[2.0, 1.0, 0.8]],
            [Some(&pi), None, Some(&pi)],
            2,
            [20.0, 15.0, 0.0],
        )
        .unwrap();
        let dir = [0.6, -0.48, 0.64];
        let at = |s: f64| [s * dir[0], s * dir[1], s * dir[2]];
        let (x1, x2) = (at(-1.3), at(0.4));
        let x0 = at(1.5);
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let t02 = model.transition(sub(x0, x2));
        for (xs, i1, i2) in [([x1, x2], 0, 1), ([x2, x1], 1, 0)] {
            assert_eq!(nearest(x0, xs.into_iter()), i2);
            let w = weights(
                &model.proportions,
                2,
                i2,
                |a, b| model.transition(sub(xs[b], xs[a])),
                |a| model.transition(sub(x0, xs[a])),
                1e-8,
            )
            .unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(w[(3 * i1 + i, j)].abs() < 1e-10, "W1 {w}");
                    assert!((w[(3 * i2 + i, j)] - t02[(i, j)]).abs() < 1e-10, "W2 {w}");
                }
            }
            for c1 in 0..3u32 {
                for c2 in 0..3u32 {
                    let mut codes = [0; 2];
                    codes[i1] = c1;
                    codes[i2] = c2;
                    let pr = combine(&w, codes.into_iter(), 3);
                    for j in 0..3 {
                        assert!((pr[j] - t02[(c2 as usize, j)]).abs() < 1e-10);
                    }
                }
            }
        }
        let xs = [x1, x2];
        let a = Mat::from_fn(6, 6, |r, c| {
            model.transition(sub(xs[c / 3], xs[r / 3]))[(r % 3, c % 3)]
        });
        let null = nalgebra::DVector::from_column_slice(&[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
        assert!((a * null).amax() < 1e-12);
    }
}
