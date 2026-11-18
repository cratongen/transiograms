//! Log-space sums, irreducibility, GTH stationary laws, matrix exponential and logarithm,
//! quasi-independence fitting and one-dimensional and simplex minimisers.

use crate::Mat;
use crate::error::{Result, convergence, invalid};

pub fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m.is_infinite() {
        return m;
    }
    m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

pub fn log_add(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi.is_infinite() {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// Strongly connected components (Tarjan, iterative) of the graph with an edge i -> j for every
/// positive off-diagonal entry, each sorted ascending, in order of their smallest member.
pub fn components(a: &Mat) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut out = Vec::new();
    let mut next = 0;
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut j)) = call.last_mut() {
            if *j < n {
                let w = *j;
                *j += 1;
                if w == v || a[(v, w)] <= 0.0 {
                    continue;
                }
                if index[w] == usize::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut comp = Vec::new();
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    comp.sort_unstable();
                    out.push(comp);
                }
            }
        }
    }
    out.sort_by_key(|c| c[0]);
    out
}

/// One strongly connected class over positive off-diagonals; a 1x1 matrix is irreducible.
pub fn is_irreducible(a: &Mat) -> bool {
    components(a).len() <= 1
}

/// Stationary law of `a = P - I` or a generator `Q` by Grassmann, Taksar and Heyman
/// elimination; only off-diagonals are read.
pub fn gth_stationary(a: &Mat) -> Result<Vec<f64>> {
    let m = a.nrows();
    if m == 0 || a.ncols() != m {
        return invalid("stationary distribution needs a non-empty square matrix");
    }
    if let Some(bad) = a.iter().find(|v| !v.is_finite()) {
        return invalid(format!(
            "stationary distribution needs finite entries, found {bad}"
        ));
    }
    let classes = components(a);
    if classes.len() > 1 {
        let listed: Vec<String> = classes.iter().map(|c| format!("{c:?}")).collect();
        return invalid(format!("reducible chain: classes {}", listed.join(", ")));
    }
    let mut w = a.clone();
    for i in 0..m {
        w[(i, i)] = 0.0;
    }
    for k in (1..m).rev() {
        let s: f64 = (0..k).map(|j| w[(k, j)]).sum();
        for i in 0..k {
            w[(i, k)] /= s;
        }
        for i in 0..k {
            let wik = w[(i, k)];
            if wik == 0.0 {
                continue;
            }
            for j in 0..k {
                if i != j {
                    w[(i, j)] += wik * w[(k, j)];
                }
            }
        }
    }
    let mut x = vec![0.0; m];
    x[0] = 1.0;
    for k in 1..m {
        x[k] = (0..k).map(|j| x[j] * w[(j, k)]).sum();
    }
    let total: f64 = x.iter().sum();
    x.iter_mut().for_each(|v| *v /= total);
    Ok(x)
}

/// `exp(Q h)` for a generator `Q`, `h >= 0`, by uniformization with scaling and squaring.
/// Every intermediate is non-negative, so the result is stochastic to rounding.
pub fn expm_rate(q: &Mat, h: f64) -> Mat {
    let n = q.nrows();
    let mu = (0..n).map(|i| q[(i, i)].abs()).fold(0.0, f64::max);
    if mu == 0.0 || h == 0.0 {
        return Mat::identity(n, n);
    }
    let r = Mat::identity(n, n) + q / mu;
    let mut s = 0u32;
    let mut lam = mu * h;
    while lam > 1.0 {
        lam /= 2.0;
        s += 1;
    }
    let mut weight = (-lam).exp();
    let mut power = Mat::identity(n, n);
    let mut t = &power * weight;
    for k in 1..=40 {
        weight *= lam / k as f64;
        power = &power * &r;
        t += &power * weight;
        if weight < 1e-18 {
            break;
        }
    }
    for _ in 0..s {
        t = &t * &t;
    }
    for mut row in t.row_iter_mut() {
        let sum: f64 = row.iter().sum();
        debug_assert!((sum - 1.0).abs() < 1e-12, "uniformization row sum {sum}");
        row /= sum;
    }
    t
}

/// General matrix exponential (nalgebra Pade, Higham 2005); a cross-check for `expm_rate`.
pub fn expm(a: &Mat) -> Mat {
    a.clone().exp()
}

fn norm1(a: &Mat) -> f64 {
    a.column_iter()
        .map(|c| c.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

const NO_LOG: &str = "transition matrix has no real logarithm";

/// Principal logarithm of a stochastic matrix by inverse scaling and squaring
/// (Denman-Beavers square roots, then the Mercator series).
pub fn logm_stochastic(p: &Mat) -> Result<Mat> {
    let n = p.nrows();
    if n == 0 || p.ncols() != n {
        return invalid("logarithm needs a non-empty square matrix");
    }
    let id = Mat::identity(n, n);
    let mut x = p.clone();
    let mut s = 0;
    while norm1(&(&x - &id)) > 0.25 {
        if s == 64 {
            return convergence(NO_LOG);
        }
        let (mut y, mut z) = (x.clone(), id.clone());
        let mut converged = false;
        for _ in 0..100 {
            let (Some(yi), Some(zi)) = (y.clone().try_inverse(), z.clone().try_inverse()) else {
                return convergence(NO_LOG);
            };
            let y_next = (&y + zi) * 0.5;
            let z_next = (&z + yi) * 0.5;
            let change = norm1(&(&y_next - &y));
            y = y_next;
            z = z_next;
            if !y.iter().all(|v| v.is_finite()) {
                return convergence(NO_LOG);
            }
            if change < 1e-14 * norm1(&y) {
                converged = true;
                break;
            }
        }
        if !converged {
            return convergence(NO_LOG);
        }
        x = y;
        s += 1;
    }
    let e = &x - &id;
    let mut term = e.clone();
    let mut log = e.clone();
    for k in 2..400 {
        term = &term * &e;
        let piece = &term * (if k % 2 == 0 { -1.0 } else { 1.0 } / k as f64);
        log += &piece;
        if norm1(&piece) < 1e-17 {
            break;
        }
    }
    if log.iter().any(|v| !v.is_finite()) {
        return convergence(NO_LOG);
    }
    Ok(log * 2f64.powi(s))
}

/// Quasi-independence table with a structural-zero diagonal: `E_ij = a_i b_j` for `i != j`,
/// `E_ii = 0`, with margins `rows` and `cols`, by iterative proportional fitting.
pub fn quasi_independence(rows: &[f64], cols: &[f64], tol: f64, max_iter: usize) -> Result<Mat> {
    let m = rows.len();
    if cols.len() != m {
        return invalid(format!(
            "{} row margins but {} column margins",
            m,
            cols.len()
        ));
    }
    if let Some(v) = rows
        .iter()
        .chain(cols)
        .find(|v| !v.is_finite() || **v < 0.0)
    {
        return invalid(format!(
            "margins must be finite and non-negative, found {v}"
        ));
    }
    let total: f64 = rows.iter().sum();
    let col_total: f64 = cols.iter().sum();
    if (total - col_total).abs() > 1e-9 * total.max(col_total) {
        return invalid(format!(
            "row margins sum to {total} but column margins to {col_total}"
        ));
    }
    let mut e = Mat::from_fn(m, m, |i, j| {
        if i != j && rows[i] > 0.0 && cols[j] > 0.0 {
            1.0
        } else {
            0.0
        }
    });
    if total == 0.0 {
        return Ok(e);
    }
    if m == 2 {
        if (rows[0] - cols[1]).abs() <= tol * total && (rows[1] - cols[0]).abs() <= tol * total {
            e[(0, 1)] = rows[0];
            e[(1, 0)] = rows[1];
            return Ok(e);
        }
        return invalid(
            "with 2 categories the margins must satisfy rows[0] = cols[1] and rows[1] = cols[0]",
        );
    }
    for i in 0..m {
        if rows[i] > 0.0 && (0..m).all(|j| e[(i, j)] == 0.0) {
            return invalid(format!(
                "row margin {i} is positive but every allowed cell is empty"
            ));
        }
        if cols[i] > 0.0 && (0..m).all(|j| e[(j, i)] == 0.0) {
            return invalid(format!(
                "column margin {i} is positive but every allowed cell is empty"
            ));
        }
    }
    for _ in 0..max_iter {
        for i in 0..m {
            let s: f64 = e.row(i).sum();
            if s > 0.0 {
                let f = rows[i] / s;
                e.row_mut(i).iter_mut().for_each(|v| *v *= f);
            }
        }
        for j in 0..m {
            let s: f64 = e.column(j).sum();
            if s > 0.0 {
                let f = cols[j] / s;
                e.column_mut(j).iter_mut().for_each(|v| *v *= f);
            }
        }
        let row_gap = (0..m)
            .map(|i| (e.row(i).sum() - rows[i]).abs())
            .fold(0.0, f64::max);
        let col_gap = (0..m)
            .map(|j| (e.column(j).sum() - cols[j]).abs())
            .fold(0.0, f64::max);
        if row_gap <= tol * total && col_gap <= tol * total {
            return Ok(e);
        }
    }
    convergence(format!(
        "quasi-independence fit did not converge in {max_iter} iterations"
    ))
}

/// Fundamental matrix `(I - P + 1 pi^T)^-1`.
pub fn fundamental_matrix(p: &Mat, pi: &[f64]) -> Result<Mat> {
    let n = p.nrows();
    if p.ncols() != n || pi.len() != n {
        return invalid("fundamental matrix needs a square P and one probability per row");
    }
    let a = Mat::from_fn(n, n, |i, j| (i == j) as u8 as f64 - p[(i, j)] + pi[j]);
    match a.lu().try_inverse() {
        Some(z) => Ok(z),
        None => invalid("I - P + 1 pi^T is singular: the chain is reducible or pi is wrong"),
    }
}

/// Minimiser of `f` on `[a, b]` by golden-section search: `(x, f(x))`.
pub fn golden_section(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> (f64, f64) {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (a.min(b), a.max(b));
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    while b - a >= tol * (0.5 * (a + b).abs() + 1e-12) {
        if fc <= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = f(d);
        }
        if b - a <= f64::EPSILON * (a.abs() + b.abs()) {
            break;
        }
    }
    let x = 0.5 * (a + b);
    (x, f(x))
}

/// Nelder-Mead simplex minimiser (reflection 1, expansion 2, contraction 0.5, shrink 0.5):
/// `(x, f(x))`.
pub fn nelder_mead(
    f: impl Fn(&[f64]) -> f64,
    x0: &[f64],
    step: f64,
    tol: f64,
    max_eval: usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let evals = std::cell::Cell::new(0usize);
    let eval = |x: &[f64]| {
        evals.set(evals.get() + 1);
        f(x)
    };
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        simplex.push(v);
    }
    let mut values: Vec<f64> = simplex.iter().map(|x| eval(x)).collect();
    loop {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&i, &j| values[i].total_cmp(&values[j]).then(i.cmp(&j)));
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        values = order.iter().map(|&i| values[i]).collect();
        let (best, worst) = (values[0], values[n]);
        if n == 0 || worst - best <= tol * (best.abs() + 1e-12) || evals.get() >= max_eval {
            return (simplex[0].clone(), best);
        }
        let centroid: Vec<f64> = (0..n)
            .map(|d| simplex[..n].iter().map(|v| v[d]).sum::<f64>() / n as f64)
            .collect();
        let toward = |coef: f64| -> Vec<f64> {
            (0..n)
                .map(|d| centroid[d] + coef * (simplex[n][d] - centroid[d]))
                .collect()
        };
        let reflected = toward(-1.0);
        let fr = eval(&reflected);
        if fr < values[0] {
            let expanded = toward(-2.0);
            let fe = eval(&expanded);
            if fe < fr {
                simplex[n] = expanded;
                values[n] = fe;
            } else {
                simplex[n] = reflected;
                values[n] = fr;
            }
        } else if fr < values[n - 1] {
            simplex[n] = reflected;
            values[n] = fr;
        } else {
            let (contracted, fc) = if fr < values[n] {
                let c = toward(-0.5);
                let v = eval(&c);
                (c, v)
            } else {
                let c = toward(0.5);
                let v = eval(&c);
                (c, v)
            };
            if fc < values[n].min(fr) {
                simplex[n] = contracted;
                values[n] = fc;
            } else {
                for i in 1..=n {
                    let shrunk: Vec<f64> = (0..n)
                        .map(|d| simplex[0][d] + 0.5 * (simplex[i][d] - simplex[0][d]))
                        .collect();
                    values[i] = eval(&shrunk);
                    simplex[i] = shrunk;
                }
            }
        }
    }
}
