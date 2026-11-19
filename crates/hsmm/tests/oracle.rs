use transiograms_core::{Error, Mat};
use transiograms_hmm::{CovarianceType, Emission, Gaussian, Sequence};
use transiograms_hsmm::{Duration, DurationDist, Hsmm};

fn gaussian(mean: f64, sd: f64) -> Emission {
    Emission::Gaussian(Gaussian {
        mean: vec![mean],
        covariance: Mat::from_diagonal_element(1, 1, sd * sd),
        covariance_type: CovarianceType::Diag,
        frozen: false,
    })
}

fn dist(law: Duration) -> DurationDist {
    DurationDist {
        law,
        frozen: false,
        pseudocount: 0.0,
    }
}

fn lse(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

fn ln_gauss(x: f64, mean: f64, sd: f64) -> f64 {
    -0.5 * ((x - mean) / sd).powi(2) - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

fn ln_fact(n: usize) -> f64 {
    (1..=n).map(|i| (i as f64).ln()).sum()
}

/// Untruncated pmf by direct closed forms, independent of the crate.
fn raw_pmf(law: &Duration, u: usize) -> f64 {
    let k = u - 1;
    match law {
        Duration::Geometric { p } => p * (1.0 - p).powi(k as i32),
        Duration::Poisson { rate } => (-rate + k as f64 * rate.ln() - ln_fact(k)).exp(),
        Duration::NegativeBinomial { r, p } => {
            let mut c = 1.0;
            for i in 0..k {
                c *= (r + i as f64) / (i as f64 + 1.0);
            }
            c * p.powf(*r) * (1.0 - p).powi(k as i32)
        }
        Duration::Empirical { pmf } => pmf.get(k).copied().unwrap_or(0.0),
    }
}

fn trunc_pmf(law: &Duration, m: usize) -> Vec<f64> {
    let raw: Vec<f64> = (1..=m).map(|u| raw_pmf(law, u)).collect();
    let s: f64 = raw.iter().sum();
    raw.iter().map(|v| v / s).collect()
}

struct Oracle {
    lnl: f64,
    best: Vec<u32>,
    best_score: f64,
    gamma: Vec<Vec<f64>>,
    start: Vec<f64>,
    xi: Vec<Vec<f64>>,
    durations: Vec<Vec<f64>>,
    entropy: f64,
}

/// Slow, obviously-correct enumeration over K^T paths with independent pmf, survivor and
/// emission densities.
fn oracle(
    start: &[f64],
    p: &Mat,
    laws: &[Duration],
    m: &[usize],
    means: &[f64],
    sd: f64,
    x: &[f64],
) -> Oracle {
    let k = start.len();
    let t_len = x.len();
    let d: Vec<Vec<f64>> = laws
        .iter()
        .zip(m)
        .map(|(l, &mm)| trunc_pmf(l, mm))
        .collect();
    let surv: Vec<Vec<f64>> = d
        .iter()
        .map(|v| (0..v.len()).map(|u| v[u..].iter().sum()).collect())
        .collect();
    let n = k.pow(t_len as u32);
    let mut scores = Vec::with_capacity(n);
    let mut paths = Vec::with_capacity(n);
    for code in 0..n {
        let mut path = vec![0usize; t_len];
        let mut c = code;
        for t in (0..t_len).rev() {
            path[t] = c % k;
            c /= k;
        }
        let mut prob = start[path[0]];
        let mut t0 = 0;
        for t in 1..=t_len {
            if t < t_len && path[t] == path[t0] {
                continue;
            }
            let (j, u) = (path[t0], t - t0);
            if u > m[j] {
                prob = 0.0;
                break;
            }
            prob *= if t == t_len {
                surv[j][u - 1]
            } else {
                d[j][u - 1] * p[(j, path[t])]
            };
            t0 = t;
        }
        let mut s = prob.ln();
        for t in 0..t_len {
            s += ln_gauss(x[t], means[path[t]], sd);
        }
        scores.push(s);
        paths.push(path);
    }
    let lnl = lse(&scores);
    let mut bi = 0;
    for i in 1..n {
        if scores[i] > scores[bi] {
            bi = i;
        }
    }
    let mut gamma = vec![vec![0.0; k]; t_len];
    let mut st = vec![0.0; k];
    let mut xi = vec![vec![0.0; k]; k];
    let mut durations: Vec<Vec<f64>> = m.iter().map(|&mm| vec![0.0; mm]).collect();
    let mut entropy = 0.0;
    for (path, &s) in paths.iter().zip(&scores) {
        if s == f64::NEG_INFINITY {
            continue;
        }
        let w = (s - lnl).exp();
        entropy -= w * (s - lnl);
        st[path[0]] += w;
        for t in 0..t_len {
            gamma[t][path[t]] += w;
            if t + 1 < t_len && path[t + 1] != path[t] {
                xi[path[t]][path[t + 1]] += w;
            }
        }
        let mut t0 = 0;
        for t in 1..=t_len {
            if t < t_len && path[t] == path[t0] {
                continue;
            }
            let (j, u) = (path[t0], t - t0);
            if t < t_len {
                durations[j][u - 1] += w;
            } else {
                for v in u..=m[j] {
                    durations[j][v - 1] += w * d[j][v - 1] / surv[j][u - 1];
                }
            }
            t0 = t;
        }
    }
    Oracle {
        lnl,
        best: paths[bi].iter().map(|&s| s as u32).collect(),
        best_score: scores[bi],
        gamma,
        start: st,
        xi,
        durations,
        entropy,
    }
}

fn check(
    start: Vec<f64>,
    p: Mat,
    laws: Vec<Duration>,
    m: Vec<usize>,
    means: Vec<f64>,
    x: Vec<f64>,
) {
    let sd = 1.3;
    let k = start.len();
    let model = Hsmm::new(
        start.clone(),
        p.clone(),
        laws.iter().cloned().map(dist).collect(),
        means.iter().map(|&mu| gaussian(mu, sd)).collect(),
        m.clone(),
    )
    .unwrap();
    let o = oracle(&start, &p, &laws, &m, &means, sd, &x);
    let seq = Sequence {
        t: x.len(),
        d: 1,
        x: x.clone(),
    };
    let post = model.posterior(&seq, None).unwrap();
    let tol = |a: f64, b: f64, e: f64| (a - b).abs() <= e * (1.0 + b.abs());
    assert!(
        tol(post.log_likelihood, o.lnl, 1e-12),
        "lnl {} vs {}",
        post.log_likelihood,
        o.lnl
    );
    assert!(tol(model.log_likelihood(&seq, None).unwrap(), o.lnl, 1e-12));
    let (path, score) = model.viterbi(&seq, None).unwrap();
    assert_eq!(path, o.best);
    assert!(tol(score, o.best_score, 1e-12));
    for t in 0..x.len() {
        for j in 0..k {
            assert!(
                (post.gamma[(t, j)] - o.gamma[t][j]).abs() < 1e-12,
                "gamma {t} {j}"
            );
        }
    }
    for i in 0..k {
        assert!((post.start[i] - o.start[i]).abs() < 1e-12);
        for j in 0..k {
            assert!((post.xi_sum[(i, j)] - o.xi[i][j]).abs() < 1e-12);
        }
        for (a, b) in post.durations[i].iter().zip(&o.durations[i]) {
            assert!((a - b).abs() < 1e-12, "durations {i}: {a} {b}");
        }
    }
    assert!(
        tol(post.entropy, o.entropy, 1e-10),
        "entropy {} vs {}",
        post.entropy,
        o.entropy
    );
}

#[test]
fn closed_form_pmfs() {
    let laws = [
        Duration::Geometric { p: 0.3 },
        Duration::Poisson { rate: 2.7 },
        Duration::NegativeBinomial { r: 2.5, p: 0.35 },
        Duration::NegativeBinomial { r: 0.4, p: 0.8 },
        Duration::Empirical {
            pmf: vec![0.2, 0.0, 0.5, 0.3],
        },
    ];
    for law in &laws {
        for m in [1usize, 3, 7, 20] {
            let a = law.log_pmf(m);
            let b = trunc_pmf(law, m);
            for u in 0..m {
                assert!((a[u].exp() - b[u]).abs() < 1e-13, "{law:?} m {m} u {u}");
            }
            let s = law.log_survivor(m);
            for u in 0..m {
                let ref_s: f64 = b[u..].iter().sum();
                assert!((s[u].exp() - ref_s).abs() < 1e-13);
            }
            let mean: f64 = b.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v).sum();
            assert!((law.mean(m) - mean).abs() < 1e-11);
        }
    }
}

#[test]
fn brute_force_two_states_short_max_duration() {
    let x = vec![0.1, 0.3, 2.9, 3.1, 3.4, 2.8, 0.2, -0.4];
    check(
        vec![0.7, 0.3],
        Mat::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]),
        vec![
            Duration::Poisson { rate: 1.4 },
            Duration::NegativeBinomial { r: 1.7, p: 0.4 },
        ],
        vec![2, 3],
        vec![0.0, 3.0],
        x,
    );
}

#[test]
fn brute_force_three_states_censored_last_long() {
    let x = vec![5.0, 0.1, 0.2, 3.2, 6.1, 6.0, 5.9, 6.2];
    check(
        vec![0.2, 0.3, 0.5],
        Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.3, 0.0, 0.7, 0.5, 0.5, 0.0]),
        vec![
            Duration::Geometric { p: 0.45 },
            Duration::Empirical {
                pmf: vec![0.0, 0.6, 0.4],
            },
            Duration::NegativeBinomial { r: 3.0, p: 0.5 },
        ],
        vec![4, 3, 2],
        vec![0.0, 3.0, 6.0],
        x,
    );
}

#[test]
fn brute_force_single_state() {
    let x = vec![0.1, -0.3, 0.5, 0.0, 0.2, -0.1];
    for m in [6usize, 9] {
        check(
            vec![1.0],
            Mat::zeros(1, 1),
            vec![Duration::Poisson { rate: 3.0 }],
            vec![m],
            vec![0.0],
            x.clone(),
        );
    }
    let model = Hsmm::new(
        vec![1.0],
        Mat::zeros(1, 1),
        vec![dist(Duration::Geometric { p: 0.2 })],
        vec![gaussian(0.0, 1.0)],
        vec![5],
    )
    .unwrap();
    let seq = Sequence {
        t: x.len(),
        d: 1,
        x,
    };
    assert!(matches!(
        model.log_likelihood(&seq, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        model.viterbi(&seq, None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        model.posterior(&seq, None),
        Err(Error::InvalidInput(_))
    ));
}
