//! Release audit: HSMM likelihood, Viterbi and posterior marginals agree with a brute-force
//! enumeration over K^T paths when rows are missing (NaN) and when states are clamped.

use transiograms_core::Mat;
use transiograms_hmm::{Clamp, CovarianceType, Emission, Gaussian, Sequence};
use transiograms_hsmm::{Duration, DurationDist, Hsmm};

const SD: f64 = 1.3;

fn ln_gauss(x: f64, mean: f64) -> f64 {
    -0.5 * ((x - mean) / SD).powi(2) - SD.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

fn trunc_pmf(law: &Duration, m: usize) -> Vec<f64> {
    let raw: Vec<f64> = (0..m)
        .map(|k| match law {
            Duration::Geometric { p } => p * (1.0 - p).powi(k as i32),
            Duration::Empirical { pmf } => pmf.get(k).copied().unwrap_or(0.0),
            _ => unreachable!(),
        })
        .collect();
    let s: f64 = raw.iter().sum();
    raw.iter().map(|v| v / s).collect()
}

struct Brute {
    lnl: f64,
    best: Vec<u32>,
    gamma: Vec<Vec<f64>>,
}

fn brute(
    start: &[f64],
    p: &Mat,
    laws: &[Duration],
    m: &[usize],
    means: &[f64],
    x: &[f64],
    allowed: Option<&[bool]>,
) -> Brute {
    let k = start.len();
    let n_t = x.len();
    let d: Vec<Vec<f64>> = laws
        .iter()
        .zip(m)
        .map(|(l, &mm)| trunc_pmf(l, mm))
        .collect();
    let surv: Vec<Vec<f64>> = d
        .iter()
        .map(|v| (0..v.len()).map(|u| v[u..].iter().sum()).collect())
        .collect();
    let mut scores = vec![];
    let mut paths = vec![];
    for code in 0..k.pow(n_t as u32) {
        let mut path = vec![0usize; n_t];
        let mut c = code;
        for t in (0..n_t).rev() {
            path[t] = c % k;
            c /= k;
        }
        let mut prob = start[path[0]];
        let mut t0 = 0;
        for t in 1..=n_t {
            if t < n_t && path[t] == path[t0] {
                continue;
            }
            let (j, u) = (path[t0], t - t0);
            if u > m[j] {
                prob = 0.0;
                break;
            }
            prob *= if t == n_t {
                surv[j][u - 1]
            } else {
                d[j][u - 1] * p[(j, path[t])]
            };
            t0 = t;
        }
        let mut s = prob.ln();
        for t in 0..n_t {
            if allowed.is_some_and(|a| !a[t * k + path[t]]) {
                s = f64::NEG_INFINITY;
            } else if !x[t].is_nan() {
                s += ln_gauss(x[t], means[path[t]]);
            }
        }
        scores.push(s);
        paths.push(path);
    }
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lnl = top + scores.iter().map(|s| (s - top).exp()).sum::<f64>().ln();
    let bi = (0..scores.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b });
    let mut gamma = vec![vec![0.0; k]; n_t];
    for (path, &s) in paths.iter().zip(&scores) {
        for t in 0..n_t {
            gamma[t][path[t]] += (s - lnl).exp();
        }
    }
    Brute {
        lnl,
        best: paths[bi].iter().map(|&s| s as u32).collect(),
        gamma,
    }
}

fn case(x: Vec<f64>, allowed: Option<Vec<bool>>) {
    let start = vec![0.2, 0.3, 0.5];
    let p = Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.3, 0.0, 0.7, 0.5, 0.5, 0.0]);
    let laws = vec![
        Duration::Geometric { p: 0.45 },
        Duration::Empirical {
            pmf: vec![0.0, 0.6, 0.4],
        },
        Duration::Geometric { p: 0.3 },
    ];
    let m = vec![4, 3, 3];
    let means = vec![0.0, 3.0, 6.0];
    let model = Hsmm::new(
        start.clone(),
        p.clone(),
        laws.iter()
            .cloned()
            .map(|law| DurationDist {
                law,
                frozen: false,
                pseudocount: 0.0,
            })
            .collect(),
        means
            .iter()
            .map(|&mu| {
                Emission::Gaussian(Gaussian {
                    mean: vec![mu],
                    covariance: Mat::from_diagonal_element(1, 1, SD * SD),
                    covariance_type: CovarianceType::Diag,
                    frozen: false,
                })
            })
            .collect(),
        m.clone(),
    )
    .unwrap();
    let b = brute(&start, &p, &laws, &m, &means, &x, allowed.as_deref());
    let seq = Sequence {
        t: x.len(),
        d: 1,
        x,
    };
    let clamp = allowed.map(|allowed| Clamp { k: 3, allowed });
    let tol = |a: f64, e: f64| (a - e).abs() <= 1e-12 * (1.0 + e.abs());
    let lnl = model.log_likelihood(&seq, clamp.as_ref()).unwrap();
    assert!(tol(lnl, b.lnl), "lnl {lnl} vs {}", b.lnl);
    let post = model.posterior(&seq, clamp.as_ref()).unwrap();
    assert!(tol(post.log_likelihood, b.lnl));
    for t in 0..seq.t {
        for j in 0..3 {
            assert!(
                (post.gamma[(t, j)] - b.gamma[t][j]).abs() < 1e-12,
                "gamma {t} {j}"
            );
        }
    }
    assert_eq!(model.viterbi(&seq, clamp.as_ref()).unwrap().0, b.best);
}

#[test]
fn missing_rows_match_brute_force() {
    let nan = f64::NAN;
    case(vec![5.0, nan, 0.2, 3.2, nan, nan, 5.9, 6.2], None);
    case(vec![nan, nan, nan, nan, nan, nan, nan], None);
}

#[test]
fn clamped_states_match_brute_force() {
    let x = vec![5.0, 0.1, 0.2, 3.2, 6.1, 6.0, 5.9, 6.2];
    let mut allowed = vec![true; x.len() * 3];
    allowed[3 * 3] = false;
    allowed[3 * 3 + 2] = false;
    allowed[6 * 3 + 2] = false;
    case(x.clone(), Some(allowed.clone()));
    let mut with_gap = x;
    with_gap[2] = f64::NAN;
    case(with_gap, Some(allowed));
}
