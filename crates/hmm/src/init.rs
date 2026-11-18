use varvite_core::{Mat, Result, Stream, invalid};

use crate::emission::{
    Categorical, CovarianceType, Density, Emission, Family, Gaussian, GaussianMixture, Poisson,
    chol_logdet,
};
use crate::model::{Hmm, Priors};
use crate::sequence::{Clamp, Sequence, check_sequence, pooled_moments};

impl Hmm {
    /// k-means (continuous) or Dirichlet / supervised (categorical) start.
    pub fn initialize(
        k: usize,
        family: &Family,
        data: &[Sequence],
        clamps: Option<&[Clamp]>,
        priors: &Priors,
        stream: &mut Stream,
    ) -> Result<Self> {
        priors.validate()?;
        check_data(k, family, data, clamps)?;
        let (_, v) = pooled_moments(data);
        let emissions = initial_emissions(k, family, data, clamps, priors.covariance, &v, stream)?;
        let (start, transition) = label_counts(&emissions, data, clamps, k);
        Hmm::new(start, transition, emissions)
    }
}

pub(crate) fn check_data(
    k: usize,
    family: &Family,
    data: &[Sequence],
    clamps: Option<&[Clamp]>,
) -> Result<()> {
    if k == 0 {
        return invalid("give at least one state");
    }
    if data.is_empty() {
        return invalid("give at least one sequence");
    }
    if let Some(c) = clamps
        && c.len() != data.len()
    {
        return invalid(format!(
            "give one clamp per sequence: {} clamps for {} sequences",
            c.len(),
            data.len()
        ));
    }
    let (d, symbols) = match *family {
        Family::Categorical { n_symbols } => {
            if n_symbols == 0 {
                return invalid("a categorical family needs at least one symbol");
            }
            (1, Some(n_symbols))
        }
        Family::Mixture { n_mix: 0, .. } => {
            return invalid("a mixture needs at least one component");
        }
        _ => (data[0].d, None),
    };
    if d == 0 {
        return invalid("sequences need at least one channel");
    }
    for (n, seq) in data.iter().enumerate() {
        check_sequence(seq, clamps.map(|c| &c[n]), n, d, k, symbols)?;
    }
    Ok(())
}

fn clamp_label(clamps: Option<&[Clamp]>, n: usize, t: usize) -> Option<usize> {
    clamps.and_then(|c| c[n].label(t))
}

/// Start and transition from per-step labels plus 1: the clamp's single allowed state, else
/// the argmax of the emission log density over allowed states (lowest index on ties).
pub(crate) fn label_counts(
    emissions: &[Emission],
    data: &[Sequence],
    clamps: Option<&[Clamp]>,
    k: usize,
) -> (Vec<f64>, Mat) {
    let mut dens: Vec<Density> = emissions.iter().map(Density::new).collect();
    let mut start = vec![1.0; k];
    let mut trans = Mat::from_element(k, k, 1.0);
    for (n, seq) in data.iter().enumerate() {
        let clamp = clamps.map(|c| &c[n]);
        let mut prev: Option<usize> = None;
        for t in 0..seq.t {
            let row = seq.row(t);
            let label = if row.iter().all(|v| v.is_nan()) {
                None
            } else {
                let (mut best, mut arg) = (f64::NEG_INFINITY, None);
                for (j, dj) in dens.iter_mut().enumerate() {
                    if clamp.is_some_and(|c| !c.allows(t, j)) {
                        continue;
                    }
                    let v = dj.log_density(row);
                    if v > best {
                        best = v;
                        arg = Some(j);
                    }
                }
                arg
            };
            if let Some(j) = label {
                if t == 0 {
                    start[j] += 1.0;
                }
                if let Some(i) = prev {
                    trans[(i, j)] += 1.0;
                }
            }
            prev = label;
        }
    }
    let s: f64 = start.iter().sum();
    start.iter_mut().for_each(|p| *p /= s);
    for i in 0..k {
        let s = trans.row(i).sum();
        trans.row_mut(i).iter_mut().for_each(|p| *p /= s);
    }
    (start, trans)
}

pub(crate) fn initial_emissions(
    k: usize,
    family: &Family,
    data: &[Sequence],
    clamps: Option<&[Clamp]>,
    eps: f64,
    v: &[f64],
    stream: &mut Stream,
) -> Result<Vec<Emission>> {
    if let Family::Categorical { n_symbols } = *family {
        let mut counts = vec![vec![0.0; n_symbols]; k];
        let mut labelled = vec![false; k];
        for (n, seq) in data.iter().enumerate() {
            for t in 0..seq.t {
                if let Some(j) = clamp_label(clamps, n, t)
                    && !seq.x[t].is_nan()
                {
                    counts[j][seq.x[t] as usize] += 1.0;
                    labelled[j] = true;
                }
            }
        }
        return Ok((0..k)
            .map(|j| {
                let w: Vec<f64> = if labelled[j] {
                    counts[j].iter().map(|c| c + 1.0).collect()
                } else {
                    (0..n_symbols).map(|_| stream.gamma(1.0)).collect()
                };
                let s: f64 = w.iter().sum();
                Emission::Categorical(Categorical {
                    probabilities: w.iter().map(|x| x / s).collect(),
                    pseudocount: 0.0,
                    frozen: false,
                })
            })
            .collect());
    }

    let d = data[0].d;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut fixed: Vec<Option<usize>> = Vec::new();
    for (n, seq) in data.iter().enumerate() {
        for t in 0..seq.t {
            if seq.row(t).iter().all(|x| !x.is_nan()) {
                rows.push(seq.row(t).to_vec());
                fixed.push(clamp_label(clamps, n, t));
            }
        }
    }
    if rows.len() < k {
        let (mean, _) = pooled_moments(data);
        rows.clear();
        fixed.clear();
        for (n, seq) in data.iter().enumerate() {
            for t in 0..seq.t {
                let r = seq.row(t);
                if r.iter().any(|x| !x.is_nan()) {
                    rows.push(
                        (0..d)
                            .map(|c| if r[c].is_nan() { mean[c] } else { r[c] })
                            .collect(),
                    );
                    fixed.push(clamp_label(clamps, n, t));
                }
            }
        }
    }
    if rows.len() < k {
        return invalid(format!(
            "{k} states need at least {k} observed steps, found {}",
            rows.len()
        ));
    }
    let all: Vec<usize> = (0..rows.len()).collect();
    let (mu, sd) = {
        let mean = col_mean(&rows, &all);
        let sd: Vec<f64> = (0..d)
            .map(|c| {
                let s = (rows.iter().map(|r| (r[c] - mean[c]).powi(2)).sum::<f64>()
                    / rows.len() as f64)
                    .sqrt();
                if s > 0.0 { s } else { 1.0 }
            })
            .collect();
        (mean, sd)
    };
    let points = Mat::from_fn(rows.len(), d, |i, c| (rows[i][c] - mu[c]) / sd[c]);
    let assign = kmeans_fixed(&points, k, &fixed, stream);
    let pooled = scatter_cov(&rows, &all, eps, v);
    let mut out = Vec::with_capacity(k);
    for j in 0..k {
        let members: Vec<usize> = all.iter().copied().filter(|&i| assign[i] == j).collect();
        out.push(match *family {
            Family::Gaussian { covariance_type } => Emission::Gaussian(gaussian_from(
                &rows,
                &members,
                covariance_type,
                eps,
                v,
                &pooled,
            )?),
            Family::Poisson => Emission::Poisson(Poisson {
                rate: col_mean(&rows, if members.is_empty() { &all } else { &members }),
                frozen: false,
            }),
            Family::Mixture {
                n_mix,
                covariance_type,
            } => {
                let idx = if members.len() >= n_mix {
                    &members
                } else {
                    &all
                };
                if idx.len() < n_mix {
                    return invalid(format!(
                        "{n_mix} mixture components need at least {n_mix} observed steps"
                    ));
                }
                let sub_points = Mat::from_fn(idx.len(), d, |i, c| points[(idx[i], c)]);
                let sub = kmeans(&sub_points, n_mix, stream);
                let mut weights = Vec::with_capacity(n_mix);
                let mut components = Vec::with_capacity(n_mix);
                for m in 0..n_mix {
                    let mem: Vec<usize> = (0..idx.len())
                        .filter(|&i| sub[i] == m)
                        .map(|i| idx[i])
                        .collect();
                    weights.push(mem.len() as f64 / idx.len() as f64);
                    components.push(gaussian_from(
                        &rows,
                        &mem,
                        covariance_type,
                        eps,
                        v,
                        &pooled,
                    )?);
                }
                Emission::Mixture(GaussianMixture {
                    weights,
                    components,
                    frozen: false,
                })
            }
            Family::Categorical { .. } => unreachable!("handled above"),
        });
    }
    Ok(out)
}

fn col_mean(rows: &[Vec<f64>], idx: &[usize]) -> Vec<f64> {
    let d = rows[0].len();
    let mut m = vec![0.0; d];
    for (n, &i) in idx.iter().enumerate() {
        for c in 0..d {
            m[c] += (rows[i][c] - m[c]) / (n + 1) as f64;
        }
    }
    m
}

/// `(C + Psi) / (n + 1)` with the prior, `C / n` without.
fn scatter_cov(rows: &[Vec<f64>], idx: &[usize], eps: f64, v: &[f64]) -> Mat {
    let d = rows[0].len();
    let mean = col_mean(rows, idx);
    let mut c = Mat::zeros(d, d);
    for &i in idx {
        for a in 0..d {
            for b in 0..d {
                c[(a, b)] += (rows[i][a] - mean[a]) * (rows[i][b] - mean[b]);
            }
        }
    }
    let n = idx.len() as f64;
    if eps > 0.0 {
        for a in 0..d {
            c[(a, a)] += eps * v[a];
        }
        c / (n + 1.0)
    } else {
        c / n.max(1.0)
    }
}

pub(crate) fn shape_cov(s: Mat, ct: CovarianceType) -> Mat {
    let d = s.nrows();
    match ct {
        CovarianceType::Full => (&s + s.transpose()) * 0.5,
        CovarianceType::Diag => Mat::from_diagonal(&s.diagonal()),
        CovarianceType::Spherical => Mat::identity(d, d) * (s.trace() / d as f64),
    }
}

fn gaussian_from(
    rows: &[Vec<f64>],
    members: &[usize],
    ct: CovarianceType,
    eps: f64,
    v: &[f64],
    pooled: &Mat,
) -> Result<Gaussian> {
    let d = rows[0].len();
    let all: Vec<usize>;
    let members = if members.is_empty() {
        all = (0..rows.len()).collect();
        &all
    } else {
        members
    };
    let cov = if members.len() < d + 1 {
        pooled.clone()
    } else {
        scatter_cov(rows, members, eps, v)
    };
    let mut cov = shape_cov(cov, ct);
    if chol_logdet(&cov).is_none() {
        let jitter = 1e-10 * cov.trace() / d as f64;
        for c in 0..d {
            cov[(c, c)] += jitter;
        }
        if jitter <= 0.0 || chol_logdet(&cov).is_none() {
            return invalid("cannot initialise a covariance: the data have no spread");
        }
    }
    Ok(Gaussian {
        mean: col_mean(rows, members),
        covariance: cov,
        covariance_type: ct,
        frozen: false,
    })
}

pub fn kmeans(points: &Mat, k: usize, stream: &mut Stream) -> Vec<usize> {
    kmeans_fixed(points, k, &vec![None; points.nrows()], stream)
}

fn dist2(points: &Mat, i: usize, c: &[f64]) -> f64 {
    c.iter()
        .enumerate()
        .map(|(a, v)| (points[(i, a)] - v).powi(2))
        .sum()
}

/// Lloyd's k-means with k-means++ seeding; rows with a fixed label keep it and seed its centre.
pub(crate) fn kmeans_fixed(
    points: &Mat,
    k: usize,
    fixed: &[Option<usize>],
    stream: &mut Stream,
) -> Vec<usize> {
    let (n, d) = points.shape();
    assert!(k >= 1 && n >= k, "kmeans needs 1 <= k <= number of points");
    let mean_of = |idx: &[usize]| -> Vec<f64> {
        (0..d)
            .map(|a| idx.iter().map(|&i| points[(i, a)]).sum::<f64>() / idx.len() as f64)
            .collect()
    };
    let mut centers: Vec<Option<Vec<f64>>> = (0..k)
        .map(|j| {
            let idx: Vec<usize> = (0..n).filter(|&i| fixed[i] == Some(j)).collect();
            (!idx.is_empty()).then(|| mean_of(&idx))
        })
        .collect();
    for j in 0..k {
        if centers[j].is_some() {
            continue;
        }
        let chosen: Vec<&Vec<f64>> = centers.iter().flatten().collect();
        let pick = if chosen.is_empty() {
            stream.below(n as u64) as usize
        } else {
            let w: Vec<f64> = (0..n)
                .map(|i| {
                    chosen
                        .iter()
                        .map(|c| dist2(points, i, c))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            if w.iter().sum::<f64>() > 0.0 {
                stream.categorical(&w)
            } else {
                stream.below(n as u64) as usize
            }
        };
        centers[j] = Some(points.row(pick).iter().copied().collect());
    }
    let mut centers: Vec<Vec<f64>> = centers.into_iter().flatten().collect();
    let mut assign = vec![usize::MAX; n];
    for _ in 0..300 {
        let new: Vec<usize> = (0..n)
            .map(|i| {
                fixed[i].unwrap_or_else(|| {
                    let mut arg = 0;
                    let mut best = dist2(points, i, &centers[0]);
                    for (j, c) in centers.iter().enumerate().skip(1) {
                        let v = dist2(points, i, c);
                        if v < best {
                            best = v;
                            arg = j;
                        }
                    }
                    arg
                })
            })
            .collect();
        if new == assign {
            break;
        }
        assign = new;
        for j in 0..k {
            let idx: Vec<usize> = (0..n).filter(|&i| assign[i] == j).collect();
            if !idx.is_empty() {
                centers[j] = mean_of(&idx);
                continue;
            }
            let mut far = None;
            let mut best = -1.0;
            for i in 0..n {
                if fixed[i].is_some() {
                    continue;
                }
                let v = dist2(points, i, &centers[assign[i]]);
                if v > best {
                    best = v;
                    far = Some(i);
                }
            }
            if let Some(i) = far {
                assign[i] = j;
                centers[j] = points.row(i).iter().copied().collect();
            }
        }
    }
    assign
}
