use varvite_core::numerics::logsumexp;
use varvite_core::{Mat, Stream};

use crate::duration::{Duration, DurationDist};
use crate::forward_backward::{backward, forward, posterior_core, sample_posterior_core};
use crate::model::Tables;
use crate::viterbi::viterbi_core;

fn random_simplex(stream: &mut Stream, k: usize, skip: Option<usize>) -> Vec<f64> {
    let mut v: Vec<f64> = (0..k)
        .map(|j| {
            if Some(j) == skip {
                0.0
            } else {
                0.2 + stream.uniform()
            }
        })
        .collect();
    let s: f64 = v.iter().sum();
    v.iter_mut().for_each(|x| *x /= s);
    v
}

fn case(seed: u64, t_len: usize) -> (Tables, Mat) {
    let mut st = Stream::new(seed, &[]);
    let k = 3;
    let start = random_simplex(&mut st, k, None);
    let mut p = Mat::zeros(k, k);
    for i in 0..k {
        let row = random_simplex(&mut st, k, Some(i));
        for j in 0..k {
            p[(i, j)] = row[j];
        }
    }
    let laws = [
        Duration::NegativeBinomial { r: 2.5, p: 0.6 },
        Duration::Poisson { rate: 2.2 },
        Duration::Empirical {
            pmf: vec![0.1, 0.3, 0.0, 0.4, 0.2, 0.1],
        },
    ];
    let refs: Vec<&Duration> = laws.iter().collect();
    let tab = Tables::new(&start, &p, &refs, &[3, 8, 5]);
    let lb = Mat::from_fn(t_len, k, |_, _| -1.0 - 2.0 * st.uniform());
    (tab, lb)
}

struct Brute {
    lnl: f64,
    best: Vec<u32>,
    best_score: f64,
    gamma: Mat,
    xi: Mat,
    start: Vec<f64>,
    durations: Vec<Vec<f64>>,
    entropy: f64,
}

/// Exhaustive search over K^T paths, run-length encoded, runs longer than `M_j` rejected.
fn brute(tab: &Tables, lb: &Mat) -> Brute {
    let (t_len, k) = (lb.nrows(), tab.k);
    let n_paths = k.pow(t_len as u32);
    let mut scores = Vec::with_capacity(n_paths);
    let mut paths = Vec::with_capacity(n_paths);
    for code in 0..n_paths {
        let mut path = vec![0usize; t_len];
        let mut c = code;
        for t in (0..t_len).rev() {
            path[t] = c % k;
            c /= k;
        }
        let mut score = tab.log_pi[path[0]];
        let mut t0 = 0;
        for t in 1..=t_len {
            if t < t_len && path[t] == path[t0] {
                continue;
            }
            let (j, u) = (path[t0], t - t0);
            if u > tab.m(j) {
                score = f64::NEG_INFINITY;
                break;
            }
            score += if t == t_len {
                tab.log_dd[j][u - 1]
            } else {
                tab.log_d[j][u - 1] + tab.log_p[(j, path[t])]
            };
            t0 = t;
        }
        for t in 0..t_len {
            score += lb[(t, path[t])];
        }
        scores.push(score);
        paths.push(path);
    }
    let lnl = logsumexp(&scores);
    let mut bi = 0;
    for i in 1..n_paths {
        if scores[i] > scores[bi] {
            bi = i;
        }
    }
    let mut gamma = Mat::zeros(t_len, k);
    let mut xi = Mat::zeros(k, k);
    let mut start = vec![0.0; k];
    let mut durations: Vec<Vec<f64>> = (0..k).map(|j| vec![0.0; tab.m(j)]).collect();
    let mut entropy = 0.0;
    for (path, &s) in paths.iter().zip(&scores) {
        if s == f64::NEG_INFINITY {
            continue;
        }
        let w = (s - lnl).exp();
        entropy -= w * (s - lnl);
        start[path[0]] += w;
        for t in 0..t_len {
            gamma[(t, path[t])] += w;
            if t + 1 < t_len && path[t + 1] != path[t] {
                xi[(path[t], path[t + 1])] += w;
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
                for v in u..=tab.m(j) {
                    durations[j][v - 1] += w * (tab.log_d[j][v - 1] - tab.log_dd[j][u - 1]).exp();
                }
            }
            t0 = t;
        }
    }
    Brute {
        lnl,
        best: paths[bi].iter().map(|&s| s as u32).collect(),
        best_score: scores[bi],
        gamma,
        xi,
        start,
        durations,
        entropy,
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

fn check_against_brute(tab: &Tables, lb: &Mat, free: &Mat) {
    let b = brute(tab, lb);
    let post = posterior_core(tab, lb, free, None).unwrap();
    assert!(
        close(post.log_likelihood, b.lnl, 1e-12),
        "{} {}",
        post.log_likelihood,
        b.lnl
    );
    let (path, score) = viterbi_core(tab, lb).unwrap();
    assert_eq!(path, b.best);
    assert!(close(score, b.best_score, 1e-12));
    for t in 0..lb.nrows() {
        let row: f64 = post.gamma.row(t).sum();
        assert!((row - 1.0).abs() < 1e-12, "row {t} sums to {row}");
        for j in 0..tab.k {
            assert!(
                (post.gamma[(t, j)] - b.gamma[(t, j)]).abs() < 1e-12,
                "gamma {t} {j}"
            );
        }
    }
    for i in 0..tab.k {
        assert!((post.start[i] - b.start[i]).abs() < 1e-12);
        for j in 0..tab.k {
            assert!((post.xi_sum[(i, j)] - b.xi[(i, j)]).abs() < 1e-12);
        }
        for (x, y) in post.durations[i].iter().zip(&b.durations[i]) {
            assert!((x - y).abs() < 1e-12, "durations {i}: {x} {y}");
        }
    }
    assert!(
        close(post.entropy, b.entropy, 1e-10),
        "{} {}",
        post.entropy,
        b.entropy
    );
}

#[test]
fn brute_force_k3_t8() {
    for seed in 0..4 {
        let (tab, lb) = case(seed, 8);
        check_against_brute(&tab, &lb, &lb);
    }
}

#[test]
fn brute_force_with_clamp() {
    let (tab, free) = case(11, 8);
    let mut lb = free.clone();
    lb[(2, 0)] = f64::NEG_INFINITY;
    lb[(2, 1)] = f64::NEG_INFINITY;
    lb[(6, 2)] = f64::NEG_INFINITY;
    check_against_brute(&tab, &lb, &free);
}

#[test]
fn forward_and_backward_agree() {
    let (tab, lb) = case(5, 120);
    let fw = forward(&tab, &lb);
    let (_, h) = backward(&tab, &lb);
    let terms: Vec<f64> = (0..tab.k).map(|j| tab.log_pi[j] + h[(0, j)]).collect();
    assert!(close(logsumexp(&terms), fw.log_likelihood, 1e-12));
    let post = posterior_core(&tab, &lb, &lb, None).unwrap();
    for t in 0..lb.nrows() {
        let row = post.gamma.row(t).sum();
        assert!((row - 1.0).abs() < 1e-12, "row {t} sums to {row}");
    }
}

#[test]
fn gamma_from_starts_and_ends_equals_segment_sums() {
    for (seed, t_len) in [(5, 120), (9, 40)] {
        let (tab, lb) = case(seed, t_len);
        let fw = forward(&tab, &lb);
        let (g, _) = backward(&tab, &lb);
        let mut gamma = Mat::zeros(t_len, tab.k);
        for end in 1..=t_len {
            for j in 0..tab.k {
                let mut e = 0.0;
                for u in 1..=end.min(tab.m(j)) {
                    e += lb[(end - u, j)];
                    let lw = if end == t_len {
                        tab.log_dd[j][u - 1]
                    } else {
                        tab.log_d[j][u - 1]
                    };
                    let eta = (fw.s[(end - u, j)] + lw + e + g[(end, j)] - fw.log_likelihood).exp();
                    for t in end - u..end {
                        gamma[(t, j)] += eta;
                    }
                }
            }
        }
        let post = posterior_core(&tab, &lb, &lb, None).unwrap();
        for t in 0..t_len {
            for j in 0..tab.k {
                let (a, b) = (post.gamma[(t, j)], gamma[(t, j)]);
                assert!((a - b).abs() < 1e-10, "seed {seed} t {t} j {j}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn impossible_sequence_names_the_step() {
    let (tab, mut lb) = case(3, 8);
    for j in 0..3 {
        lb[(4, j)] = f64::NEG_INFINITY;
    }
    let err = posterior_core(&tab, &lb, &lb, Some(7)).unwrap_err();
    assert!(err.to_string().contains("sequence 7, step 4"), "{err}");
    assert!(viterbi_core(&tab, &lb).is_err());
}

/// Log-space HMM forward and Viterbi, the reference for geometric equivalence.
fn hmm_reference(start: &[f64], a: &Mat, lb: &Mat) -> (f64, Vec<u32>) {
    let (t_len, k) = (lb.nrows(), start.len());
    let mut alpha: Vec<f64> = (0..k).map(|j| start[j].ln() + lb[(0, j)]).collect();
    let mut delta = alpha.clone();
    let mut psi = vec![vec![0usize; k]; t_len];
    for t in 1..t_len {
        let mut na = vec![0.0; k];
        let mut nd = vec![0.0; k];
        for j in 0..k {
            let terms: Vec<f64> = (0..k).map(|i| alpha[i] + a[(i, j)].ln()).collect();
            na[j] = logsumexp(&terms) + lb[(t, j)];
            let mut bi = 0;
            for i in 1..k {
                if delta[i] + a[(i, j)].ln() > delta[bi] + a[(bi, j)].ln() {
                    bi = i;
                }
            }
            psi[t][j] = bi;
            nd[j] = delta[bi] + a[(bi, j)].ln() + lb[(t, j)];
        }
        alpha = na;
        delta = nd;
    }
    let mut s = 0;
    for j in 1..k {
        if delta[j] > delta[s] {
            s = j;
        }
    }
    let mut path = vec![0u32; t_len];
    for t in (0..t_len).rev() {
        path[t] = s as u32;
        s = psi[t][s];
    }
    (logsumexp(&alpha), path)
}

#[test]
fn geometric_hsmm_equals_hmm() {
    let mut st = Stream::new(42, &[]);
    let k = 3;
    let a = Mat::from_row_slice(k, k, &[0.9, 0.06, 0.04, 0.1, 0.85, 0.05, 0.02, 0.18, 0.8]);
    let start = vec![0.5, 0.3, 0.2];
    let lb = Mat::from_fn(150, k, |_, _| -3.0 * st.uniform());
    let m = 400;
    assert!(0.9f64.powi(m as i32) < 1e-14);
    let mut p = Mat::zeros(k, k);
    let mut laws = Vec::new();
    for j in 0..k {
        for i in 0..k {
            if i != j {
                p[(j, i)] = a[(j, i)] / (1.0 - a[(j, j)]);
            }
        }
        laws.push(Duration::Geometric { p: 1.0 - a[(j, j)] });
    }
    let refs: Vec<&Duration> = laws.iter().collect();
    let tab = Tables::new(&start, &p, &refs, &[m; 3]);
    let (lnl_hmm, path_hmm) = hmm_reference(&start, &a, &lb);
    let fw = forward(&tab, &lb);
    assert!(
        close(fw.log_likelihood, lnl_hmm, 1e-10),
        "{} {lnl_hmm}",
        fw.log_likelihood
    );
    assert_eq!(viterbi_core(&tab, &lb).unwrap().0, path_hmm);
}

#[test]
fn posterior_samples_follow_brute_force_marginals() {
    let (tab, lb) = case(9, 8);
    let b = brute(&tab, &lb);
    let n = 20_000;
    let mut st = Stream::new(1, &[]);
    let draws = sample_posterior_core(&tab, &lb, n, &mut st).unwrap();
    for t in 0..8 {
        for j in 0..3 {
            let f = draws.iter().filter(|d| d[t] == j as u32).count() as f64 / n as f64;
            let p = b.gamma[(t, j)];
            let sd = (p * (1.0 - p) / n as f64).sqrt();
            assert!((f - p).abs() <= 4.0 * sd + 1e-9, "t {t} j {j}: {f} vs {p}");
        }
    }
    for d in &draws {
        let mut t0 = 0;
        for t in 1..=8 {
            if t < 8 && d[t] == d[t0] {
                continue;
            }
            assert!(t - t0 <= tab.m(d[t0] as usize));
            t0 = t;
        }
    }
}

#[test]
fn duration_laws_are_normalised() {
    let laws = [
        Duration::Geometric { p: 0.3 },
        Duration::Poisson { rate: 3.5 },
        Duration::NegativeBinomial { r: 0.7, p: 0.2 },
        Duration::Empirical {
            pmf: vec![1.0, 2.0, 0.0, 1.0],
        },
    ];
    for law in &laws {
        let lp = law.log_pmf(12);
        let total: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-14);
        let ls = law.log_survivor(12);
        assert!(ls[0].abs() < 1e-14);
        for u in 0..11 {
            let d = ls[u].exp() - ls[u + 1].exp();
            assert!((d - lp[u].exp()).abs() < 1e-14);
        }
    }
    let g = Duration::Geometric { p: 0.3 };
    assert!(close(g.mass_above(10), 0.7f64.powi(10), 1e-14));
    assert!(close(g.mean(2000), 1.0 / 0.3, 1e-12));
    let pois = Duration::Poisson { rate: 3.5 };
    assert!(close(pois.mean(200), 4.5, 1e-12));
    let nb = Duration::NegativeBinomial { r: 2.0, p: 0.4 };
    assert!(close(nb.mean(500), 1.0 + 2.0 * 0.6 / 0.4, 1e-12));
    let e = Duration::Empirical {
        pmf: vec![1.0, 2.0, 0.0, 1.0],
    };
    assert!(close(e.mass_above(2), 0.25, 1e-14));
    assert_eq!(e.n_params(4), 3);
}

#[test]
fn duration_m_step_closed_forms() {
    let counts = [3.0, 5.5, 2.0, 0.0, 1.25, 0.5];
    let total: f64 = counts.iter().sum();
    let mbar: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, c)| (i + 1) as f64 * c)
        .sum::<f64>()
        / total;
    let mut padded = counts.to_vec();
    padded.resize(400, 0.0);
    let Duration::Geometric { p } = Duration::Geometric { p: 0.5 }.fit(&padded, 0.0).unwrap()
    else {
        panic!()
    };
    assert!(close(1.0 / p, mbar, 1e-14));
    let Duration::Poisson { rate } = Duration::Poisson { rate: 1.0 }.fit(&padded, 0.0).unwrap()
    else {
        panic!()
    };
    assert!(close(1.0 + rate, mbar, 1e-14));
    let m = counts.len();
    for template in [
        Duration::Geometric { p: 0.5 },
        Duration::Poisson { rate: 1.0 },
    ] {
        let law = template.fit(&counts, 0.0).unwrap();
        assert!(close(law.mean(m), mbar, 1e-10), "{law:?}");
        let s0 = law.weighted_log_lik(&counts, m);
        let nudged = |f: f64| match law {
            Duration::Geometric { p } => Duration::Geometric { p: p * f },
            Duration::Poisson { rate } => Duration::Poisson { rate: rate * f },
            _ => unreachable!(),
        };
        for f in [1.0 - 1e-3, 1.0 + 1e-3] {
            assert!(nudged(f).weighted_log_lik(&counts, m) <= s0, "{law:?} {f}");
        }
    }
    let nb = Duration::NegativeBinomial { r: 1.0, p: 0.5 }
        .fit(&counts, 0.0)
        .unwrap();
    let Duration::NegativeBinomial { r, p } = nb else {
        panic!()
    };
    let s0 = nb.weighted_log_lik(&counts, m);
    for fr in [1.0 - 1e-3, 1.0, 1.0 + 1e-3] {
        for fp in [1.0 - 1e-3, 1.0, 1.0 + 1e-3] {
            let other = Duration::NegativeBinomial {
                r: r * fr,
                p: (p * fp).min(1.0),
            };
            assert!(other.weighted_log_lik(&counts, m) <= s0, "{fr} {fp}");
        }
    }
    let Duration::NegativeBinomial { r, .. } = Duration::NegativeBinomial { r: 1.0, p: 0.5 }
        .fit(&padded, 0.0)
        .unwrap()
    else {
        panic!()
    };
    let score = |r: f64| {
        Duration::NegativeBinomial {
            r,
            p: r / (r + mbar - 1.0),
        }
        .weighted_log_lik(&padded, padded.len())
    };
    let s0 = score(r);
    assert!(score(r * (1.0 + 1e-3)) <= s0 && score(r * (1.0 - 1e-3)) <= s0);
    let point = Duration::NegativeBinomial { r: 1.0, p: 0.5 }
        .fit(&[4.0, 0.0], 0.0)
        .unwrap();
    assert_eq!(point, Duration::NegativeBinomial { r: 1.0, p: 1.0 });
    let Duration::Empirical { pmf } = Duration::Empirical { pmf: vec![1.0] }
        .fit(&counts, 0.5)
        .unwrap()
    else {
        panic!()
    };
    for (u, c) in counts.iter().enumerate() {
        assert!(close(pmf[u], (c + 0.5) / (total + 3.0), 1e-14));
    }
    assert!(
        Duration::Geometric { p: 0.5 }
            .fit(&[0.0, 0.0], 0.0)
            .is_err()
    );
}

#[test]
fn duration_parameters_recovered_from_seeded_samples() {
    let n = 40_000;
    let m = 300;
    let tally = |law: &Duration, key: u64| -> Vec<f64> {
        let lp = law.log_pmf(m);
        let mut st = Stream::new(7, &[key]);
        let mut counts = vec![0.0; m];
        for _ in 0..n {
            counts[Duration::draw(&lp, &mut st) - 1] += 1.0;
        }
        counts
    };
    let geo = Duration::Geometric { p: 0.2 };
    let Duration::Geometric { p } = geo.fit(&tally(&geo, 0), 0.0).unwrap() else {
        panic!()
    };
    let sd = 0.2 * (0.8 / n as f64).sqrt();
    assert!((p - 0.2).abs() < 4.0 * sd, "{p}");
    let pois = Duration::Poisson { rate: 4.0 };
    let Duration::Poisson { rate } = pois.fit(&tally(&pois, 1), 0.0).unwrap() else {
        panic!()
    };
    assert!((rate - 4.0).abs() < 4.0 * (4.0 / n as f64).sqrt(), "{rate}");
    let nb = Duration::NegativeBinomial { r: 3.0, p: 0.4 };
    let Duration::NegativeBinomial { r, p } = nb.fit(&tally(&nb, 2), 0.0).unwrap() else {
        panic!()
    };
    assert!((r - 3.0).abs() < 0.3 && (p - 0.4).abs() < 0.03, "{r} {p}");
    let emp = Duration::Empirical {
        pmf: vec![0.1, 0.4, 0.3, 0.2],
    };
    let counts: Vec<f64> = tally(&emp, 3).into_iter().take(4).collect();
    let Duration::Empirical { pmf } = emp.fit(&counts, 0.0).unwrap() else {
        panic!()
    };
    for (a, b) in pmf.iter().zip([0.1, 0.4, 0.3, 0.2]) {
        assert!((a - b).abs() < 4.0 * (b * (1.0 - b) / n as f64).sqrt());
    }
}

#[test]
fn duration_dist_round_trips_through_json() {
    let d = DurationDist {
        law: Duration::NegativeBinomial { r: 2.0, p: 0.3 },
        frozen: true,
        pseudocount: 0.5,
    };
    let s = serde_json::to_string(&d).unwrap();
    assert!(s.contains("\"law\":\"negative_binomial\""));
    assert_eq!(serde_json::from_str::<DurationDist>(&s).unwrap(), d);
}
