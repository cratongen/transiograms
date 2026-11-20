use proptest::prelude::*;
use transiograms_core::numerics::*;
use transiograms_core::synthetic::*;
use transiograms_core::*;

fn golden_q() -> Mat {
    Mat::from_row_slice(
        3,
        3,
        &[-0.25, 0.15, 0.10, 0.35, -0.5, 0.15, 0.1, 0.375, -0.475],
    )
}

fn max_diff(a: &Mat, b: &Mat) -> f64 {
    (a - b).iter().fold(0.0, |m, v| m.max(v.abs()))
}

#[test]
fn logsumexp_handles_infinities() {
    let ninf = f64::NEG_INFINITY;
    assert_eq!(logsumexp(&[ninf, ninf]), ninf);
    assert_eq!(logsumexp(&[]), ninf);
    assert_eq!(logsumexp(&[ninf, f64::INFINITY]), f64::INFINITY);
    assert!((logsumexp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-12);
    assert!((logsumexp(&[0.0, ninf, 0.0]) - 2f64.ln()).abs() < 1e-15);
    assert_eq!(log_add(ninf, ninf), ninf);
    assert!((log_add(-1.0, -2.0) - logsumexp(&[-1.0, -2.0])).abs() < 1e-15);
}

#[test]
fn gth_golden_and_properties() {
    let p = gth_stationary(&golden_q()).unwrap();
    for (a, b) in p.iter().zip([0.5, 0.3, 0.2]) {
        assert!((a - b).abs() < 1e-14, "{p:?}");
    }
    let t = TransitionMatrix::new(expm_rate(&golden_q(), 0.7)).unwrap();
    let pi = t.stationary().unwrap();
    let pi_p: Vec<f64> = (0..3)
        .map(|j| (0..3).map(|i| pi[i] * t.matrix()[(i, j)]).sum())
        .collect();
    for j in 0..3 {
        assert!((pi_p[j] - pi[j]).abs() < 1e-12);
    }
    let reducible = Mat::from_row_slice(3, 3, &[0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.3, 0.7]);
    let e = TransitionMatrix::new(reducible)
        .unwrap()
        .stationary()
        .unwrap_err()
        .to_string();
    assert!(e.contains("reducible chain: classes [0, 1], [2]"), "{e}");
    let periodic = TransitionMatrix::new(Mat::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0])).unwrap();
    assert_eq!(periodic.stationary().unwrap(), vec![0.5, 0.5]);
    assert_eq!(gth_stationary(&Mat::zeros(1, 1)).unwrap(), vec![1.0]);
    assert!(is_irreducible(&golden_q()));
}

#[test]
fn expm_rate_golden_values() {
    let q = golden_q();
    let golden = [
        (
            0.5,
            [
                0.8894003915,
                0.0663597651,
                0.0442398434,
                0.1474291504,
                0.7899162352,
                0.0626546144,
                0.0553552955,
                0.1492262345,
                0.79541847,
            ],
        ),
        (
            1.0,
            [
                0.8032653299,
                0.1180408021,
                0.0786938681,
                0.2510484883,
                0.6431007346,
                0.1058507771,
                0.115263943,
                0.2402468928,
                0.6444891642,
            ],
        ),
        (
            5.0,
            [
                0.5410424993,
                0.2753745004,
                0.1835830003,
                0.483595679,
                0.3205022315,
                0.1959020895,
                0.4220002332,
                0.3308104017,
                0.2471893651,
            ],
        ),
    ];
    for (h, values) in golden {
        let t = expm_rate(&q, h);
        assert!(
            max_diff(&t, &Mat::from_row_slice(3, 3, &values)) < 1e-9,
            "T({h}) = {t}"
        );
    }
    let t50 = expm_rate(&q, 50.0);
    for i in 0..3 {
        for (j, p) in [0.5, 0.3, 0.2].iter().enumerate() {
            assert!((t50[(i, j)] - p).abs() < 1e-9);
        }
    }
}

#[test]
fn expm_rate_properties() {
    let q = golden_q();
    assert_eq!(expm_rate(&q, 0.0), Mat::identity(3, 3));
    for h in [0.01, 0.3, 1.7, 12.0] {
        let t = expm_rate(&q, h);
        assert!(t.iter().all(|&v| v >= 0.0));
        for i in 0..3 {
            assert!((t.row(i).sum() - 1.0).abs() < 1e-12);
        }
        assert!(
            max_diff(&t, &expm(&(&q * h))) < 1e-12,
            "Pade differs at h = {h}"
        );
    }
    for (a, b) in [(0.3, 0.9), (1.0, 2.5), (0.05, 7.0)] {
        let lhs = expm_rate(&q, a + b);
        let rhs = expm_rate(&q, a) * expm_rate(&q, b);
        assert!(max_diff(&lhs, &rhs) < 1e-12);
    }
    let far = expm_rate(&q, 1e4 / 0.5);
    assert!(far.iter().all(|v| v.is_finite()));
    assert!((far[(1, 0)] - 0.5).abs() < 1e-10);
}

#[test]
fn expm_rate_stays_stochastic_at_huge_lags() {
    let q = RateMatrix::max_entropy(&[0.5, 0.3, 0.2], &[0.4, 0.1, 0.05])
        .unwrap()
        .matrix()
        .clone();
    let pi = gth_stationary(&q).unwrap();
    for h in [1e3, 1e4, 1e6, 1e9] {
        let t = expm_rate(&q, h);
        for i in 0..3 {
            assert!((t.row(i).sum() - 1.0).abs() < 1e-12, "h = {h}");
            for j in 0..3 {
                assert!((t[(i, j)] - pi[j]).abs() < 1e-10, "h = {h}: {t}");
            }
        }
    }
}

#[test]
fn expm_rate_two_state_closed_form() {
    let (a, b) = (0.7, 0.2);
    let s = a + b;
    let q = Mat::from_row_slice(2, 2, &[-a, a, b, -b]);
    for h in [0.0, 0.1, 1.0, 3.3, 40.0] {
        let t = expm_rate(&q, h);
        let e = (-s * h).exp();
        let t00 = b / s + (a / s) * e;
        let t01 = (a / s) * (1.0 - e);
        let t10 = (b / s) * (1.0 - e);
        let t11 = a / s + (b / s) * e;
        let exact = Mat::from_row_slice(2, 2, &[t00, t01, t10, t11]);
        assert!(max_diff(&t, &exact) < 1e-14, "h = {h}");
    }
}

#[test]
fn expm_rate_single_rate_repeated_eigenvalue() {
    let p = [0.1, 0.2, 0.3, 0.4];
    let lambda = 0.8;
    let q = Mat::from_fn(4, 4, |i, j| lambda * (p[j] - (i == j) as u8 as f64));
    let rm = RateMatrix::new(q.clone()).unwrap();
    for (i, l) in rm.mean_lengths().iter().enumerate() {
        assert!((l - 1.0 / (lambda * (1.0 - p[i]))).abs() < 1e-14);
    }
    for h in [0.2, 1.0, 6.0] {
        let e = (-lambda * h).exp();
        let exact = Mat::from_fn(4, 4, |i, j| p[j] + ((i == j) as u8 as f64 - p[j]) * e);
        assert!(max_diff(&expm_rate(&q, h), &exact) < 1e-13);
    }
}

#[test]
fn logm_inverts_expm_on_the_golden_q() {
    let q = golden_q();
    for h in [0.5, 1.0, 2.0] {
        let log = logm_stochastic(&expm_rate(&q, h)).unwrap() / h;
        assert!(max_diff(&log, &q) < 1e-10, "h = {h}: {log}");
        let (r, proj) =
            RateMatrix::from_transition(&TransitionMatrix::new(expm_rate(&q, h)).unwrap(), h)
                .unwrap();
        assert!(max_diff(r.matrix(), &q) < 1e-10);
        assert!(proj.distance < 1e-10);
    }
    let swap = Mat::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
    assert!(matches!(logm_stochastic(&swap), Err(Error::Convergence(_))));
}

#[test]
fn from_transition_projects_to_a_generator() {
    let p = TransitionMatrix::new(Mat::from_row_slice(
        3,
        3,
        &[0.6, 0.4, 0.0, 0.0, 0.6, 0.4, 0.4, 0.0, 0.6],
    ))
    .unwrap();
    let (r, proj) = RateMatrix::from_transition(&p, 1.0).unwrap();
    assert!(!proj.exact && proj.distance > 0.0);
    assert!(RateMatrix::new(r.matrix().clone()).is_ok());
}

#[test]
fn quasi_independence_golden_tables() {
    let n = Mat::from_row_slice(
        4,
        4,
        &[
            0., 30., 10., 5., 8., 0., 25., 12., 20., 6., 0., 9., 10., 14., 8., 0.,
        ],
    );
    let r: Vec<f64> = (0..4).map(|i| n.row(i).sum()).collect();
    let c: Vec<f64> = (0..4).map(|j| n.column(j).sum()).collect();
    let e = quasi_independence(&r, &c, 1e-13, 10_000).unwrap();
    let golden = Mat::from_row_slice(
        4,
        4,
        &[
            0.0,
            20.6805682797,
            15.4762764461,
            8.8431552742,
            17.022383202,
            0.0,
            17.8042536869,
            10.1733631111,
            11.68497564,
            16.3315427453,
            0.0,
            6.9834816147,
            9.292641158,
            12.987888975,
            9.719469867,
            0.0,
        ],
    );
    assert!(max_diff(&e, &golden) < 1e-8, "{e}");
    for i in 0..4 {
        assert!((e.row(i).sum() - r[i]).abs() < 1e-12 * 1000.0);
        assert!((e.column(i).sum() - c[i]).abs() < 1e-12 * 1000.0);
    }
    // Quasi-independence: every cross-product ratio over off-diagonal cells is 1.
    let ratio = e[(0, 1)] * e[(2, 3)] / (e[(0, 3)] * e[(2, 1)]);
    assert!((ratio - 1.0).abs() < 1e-10);
    let ratio = e[(1, 0)] * e[(3, 2)] / (e[(1, 2)] * e[(3, 0)]);
    assert!((ratio - 1.0).abs() < 1e-10);

    let n3 = Mat::from_row_slice(3, 3, &[0., 12., 7., 5., 0., 14., 9., 4., 0.]);
    let r: Vec<f64> = (0..3).map(|i| n3.row(i).sum()).collect();
    let c: Vec<f64> = (0..3).map(|j| n3.column(j).sum()).collect();
    let e = quasi_independence(&r, &c, 1e-13, 10_000).unwrap();
    let golden = Mat::from_row_slice(
        3,
        3,
        &[
            0.0,
            8.919987347,
            10.080012653,
            8.080012653,
            0.0,
            10.919987347,
            5.919987347,
            7.080012653,
            0.0,
        ],
    );
    assert!(max_diff(&e, &golden) < 1e-8, "{e}");
    let cycle = e[(0, 1)] * e[(1, 2)] * e[(2, 0)] / (e[(0, 2)] * e[(2, 1)] * e[(1, 0)]);
    assert!((cycle - 1.0).abs() < 1e-10);

    assert!(quasi_independence(&[3.0, 2.0], &[2.0, 3.0], 1e-13, 100).is_ok());
    assert!(quasi_independence(&[3.0, 2.0], &[3.0, 2.0], 1e-13, 100).is_err());
}

#[test]
fn carle_background_fill_golden() {
    let mut partial = Mat::zeros(3, 3);
    partial[(0, 0)] = -0.25;
    partial[(0, 1)] = 0.6 / 4.0;
    partial[(1, 0)] = 0.7 / 2.0;
    partial[(1, 1)] = -0.5;
    let r = RateMatrix::from_partial(&partial, &[0.5, 0.3, 0.2], 2).unwrap();
    assert!(max_diff(r.matrix(), &golden_q()) < 1e-12, "{}", r.matrix());
    assert!((r.mean_lengths()[2] - 2.105263157894737).abs() < 1e-12);
    let pi = r.embedded();
    let golden_pi = Mat::from_row_slice(
        3,
        3,
        &[
            0.0,
            0.6,
            0.4,
            0.7,
            0.0,
            0.3,
            0.2105263158,
            0.7894736842,
            0.0,
        ],
    );
    assert!(max_diff(pi.matrix(), &golden_pi) < 1e-9);
    let nu = pi.stationary().unwrap();
    for (a, b) in nu.iter().zip([0.3378378378, 0.4054054054, 0.2567567568]) {
        assert!((a - b).abs() < 1e-9);
    }
    let eig = r.matrix().clone().complex_eigenvalues();
    let mut re: Vec<f64> = eig.iter().map(|z| z.re).collect();
    re.sort_by(f64::total_cmp);
    for (a, b) in re.iter().zip([-0.725, -0.5, 0.0]) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(!r.is_reversible(&[0.5, 0.3, 0.2], 1e-9));
    let back = RateMatrix::from_embedded(&pi, &r.mean_lengths()).unwrap();
    assert!(max_diff(back.matrix(), r.matrix()) < 1e-14);

    partial[(0, 1)] = 0.4;
    let e = RateMatrix::from_partial(&partial, &[0.5, 0.3, 0.2], 2)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("rates incompatible with proportions: entry (0, 2)"),
        "{e}"
    );
}

#[test]
fn maximum_entropy_rates() {
    let p = [0.5, 0.3, 0.2];
    let l = [4.0, 2.0, 3.0];
    let r = RateMatrix::max_entropy(&p, &l).unwrap();
    let scale = r.matrix().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for j in 0..3 {
        let s: f64 = (0..3).map(|i| p[i] * r.matrix()[(i, j)]).sum();
        assert!(s.abs() < 1e-12 * scale);
    }
    for (a, b) in r.mean_lengths().iter().zip(l) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(r.is_reversible(&p, 1e-12));
    let st = r.stationary().unwrap();
    for (a, b) in st.iter().zip(p) {
        assert!((a - b).abs() < 1e-12);
    }
    let e = RateMatrix::max_entropy(&[0.8, 0.1, 0.1], &[1.0, 1.0, 1.0])
        .unwrap_err()
        .to_string();
    assert!(e.contains("category 0"), "{e}");
    let two = RateMatrix::max_entropy(&[0.6, 0.4], &[3.0, 2.0]).unwrap();
    assert!((two.matrix()[(0, 1)] - 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn matrix_validation() {
    assert!(TransitionMatrix::new(Mat::from_row_slice(2, 2, &[0.5, 0.5, 0.2, 0.7])).is_err());
    assert!(TransitionMatrix::new(Mat::from_row_slice(2, 2, &[1.1, -0.1, 0.5, 0.5])).is_err());
    assert!(TransitionMatrix::new(Mat::from_row_slice(1, 2, &[0.5, 0.5])).is_err());
    assert!(TransitionMatrix::new(Mat::from_row_slice(2, 2, &[f64::NAN, 1.0, 0.5, 0.5])).is_err());
    assert!(RateMatrix::new(Mat::from_row_slice(2, 2, &[-1.0, 1.0, 1.0, -0.5])).is_err());
    assert!(RateMatrix::new(Mat::from_row_slice(2, 2, &[0.0, 0.0, 1.0, -1.0])).is_err());
    assert!(RateMatrix::new(Mat::from_row_slice(2, 2, &[1.0, -1.0, 1.0, -1.0])).is_err());
    assert!(RateMatrix::new(Mat::zeros(1, 1)).is_ok());
    let t = TransitionMatrix::new(Mat::from_row_slice(2, 2, &[0.9, 0.1, 0.3, 0.7])).unwrap();
    assert!(max_diff(&t.power(5), &(t.matrix().pow(5))) < 1e-15);
    assert_eq!(t.power(0), Mat::identity(2, 2));
    let pi = t.stationary().unwrap();
    let rev = t.reversed(&pi).unwrap();
    assert!(
        max_diff(rev.matrix(), t.matrix()) < 1e-15,
        "two-state chains are reversible"
    );
    assert!(!t.is_embedded());
}

#[test]
fn transition_from_counts() {
    let counts = Mat::from_row_slice(3, 3, &[0., 3., 1., 0., 0., 0., 2., 2., 0.]);
    let e = TransitionMatrix::from_counts(&counts, 0.0, true, EmptyRow::Raise)
        .unwrap_err()
        .to_string();
    assert!(e.contains("category 1"), "{e}");
    let u = TransitionMatrix::from_counts(&counts, 0.0, true, EmptyRow::Uniform).unwrap();
    assert_eq!(
        u.matrix().row(1).iter().copied().collect::<Vec<_>>(),
        vec![0.5, 0.0, 0.5]
    );
    assert_eq!(u.matrix()[(0, 1)], 0.75);
    assert!(u.is_embedded());
    let a = TransitionMatrix::from_counts(&counts, 1.0, false, EmptyRow::Raise).unwrap();
    assert_eq!(a.matrix()[(0, 0)], 1.0 / 7.0);
    assert_eq!(a.matrix()[(1, 1)], 1.0 / 3.0);
    let diag = Mat::from_row_slice(2, 2, &[1., 1., 1., 0.]);
    assert!(TransitionMatrix::from_counts(&diag, 0.0, true, EmptyRow::Raise).is_err());
}

#[test]
fn fundamental_matrix_identities() {
    let t = TransitionMatrix::new(expm_rate(&golden_q(), 1.0)).unwrap();
    let pi = t.stationary().unwrap();
    let z = fundamental_matrix(t.matrix(), &pi).unwrap();
    let a = Mat::from_fn(3, 3, |i, j| {
        (i == j) as u8 as f64 - t.matrix()[(i, j)] + pi[j]
    });
    assert!(max_diff(&(&a * &z), &Mat::identity(3, 3)) < 1e-12);
    for j in 0..3 {
        let s: f64 = (0..3).map(|i| pi[i] * z[(i, j)]).sum();
        assert!((s - pi[j]).abs() < 1e-12);
    }
}

#[test]
fn minimisers() {
    let (x, f) = golden_section(|x| (x - 2.0).powi(2) + 1.0, 0.0, 5.0, 1e-10);
    assert!((x - 2.0).abs() < 1e-6 && (f - 1.0).abs() < 1e-12);
    let rosen = |v: &[f64]| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2);
    let (x, f, converged) = nelder_mead(rosen, &[-1.2, 1.0], 0.5, 1e-16, 10_000);
    assert!(
        converged && (x[0] - 1.0).abs() < 1e-4 && (x[1] - 1.0).abs() < 1e-4 && f < 1e-8,
        "{x:?} {f}"
    );
    let (_, _, converged) = nelder_mead(rosen, &[-1.2, 1.0], 0.5, 1e-16, 20);
    assert!(!converged, "20 evaluations cannot meet tol 1e-16 on Rosenbrock");
}

#[test]
fn synthetic_markov_recovers_its_matrix() {
    let p = TransitionMatrix::new(expm_rate(&golden_q(), 1.0)).unwrap();
    let holes = markov_holes(&p, &[50_000, 50_000], 11).unwrap();
    let mut counts = Mat::zeros(3, 3);
    for h in &holes {
        for w in h.windows(2) {
            counts[(w[0] as usize, w[1] as usize)] += 1.0;
        }
    }
    for i in 0..3 {
        let n = counts.row(i).sum();
        for j in 0..3 {
            let pij = p.matrix()[(i, j)];
            let sd = (pij * (1.0 - pij) / n).sqrt();
            assert!((counts[(i, j)] / n - pij).abs() < 4.0 * sd, "({i}, {j})");
        }
    }
}

#[test]
fn synthetic_rate_holes_recover_lengths_and_rates() {
    let q = RateMatrix::new(golden_q()).unwrap();
    let lengths = vec![500.0; 200];
    let iv = rate_holes(&q, &lengths, Direction::Up, 3).unwrap();
    let seqs = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
    assert_eq!(seqs.holes.len(), 200);
    let mut exposure = [0.0; 3];
    let mut n = Mat::zeros(3, 3);
    for beds in seqs.beds(Direction::Up, 1e-9, true) {
        for (i, b) in beds.iter().enumerate() {
            exposure[b.code as usize] += b.thickness;
            if let Some(next) = beds.get(i + 1) {
                n[(b.code as usize, next.code as usize)] += 1.0;
            }
        }
    }
    for j in 0..3 {
        for k in (0..3).filter(|&k| k != j) {
            let rate = n[(j, k)] / exposure[j];
            let truth = golden_q()[(j, k)];
            let sd = (truth / exposure[j]).sqrt();
            assert!(
                (rate - truth).abs() < 4.0 * sd,
                "q[{j},{k}] = {rate}, truth {truth}"
            );
        }
    }
    let first_hole_top = iv.from[0];
    assert_eq!(first_hole_top, 0.0);
}

#[test]
fn synthetic_hmm_sample_follows_its_emitters() {
    let a = TransitionMatrix::new(Mat::from_row_slice(2, 2, &[0.95, 0.05, 0.1, 0.9])).unwrap();
    let emitters = [
        Emitter::Gaussian {
            mean: vec![0.0, 10.0],
            sd: vec![1.0, 2.0],
        },
        Emitter::Gaussian {
            mean: vec![5.0, -3.0],
            sd: vec![0.5, 1.0],
        },
    ];
    let s = hmm_sample(&[1.0, 0.0], &a, &emitters, &[30_000, 20_000], 9).unwrap();
    assert_eq!(s.d, 2);
    assert_eq!(s.states[0][0], 0);
    let mut sum = [[0.0; 2]; 2];
    let mut count = [0.0; 2];
    for (states, obs) in s.states.iter().zip(&s.observations) {
        assert_eq!(obs.len(), states.len() * 2);
        for (t, &st) in states.iter().enumerate() {
            count[st as usize] += 1.0;
            sum[st as usize][0] += obs[2 * t];
            sum[st as usize][1] += obs[2 * t + 1];
        }
    }
    for (st, e) in emitters.iter().enumerate() {
        let Emitter::Gaussian { mean, sd } = e else {
            unreachable!()
        };
        for c in 0..2 {
            let m = sum[st][c] / count[st];
            assert!((m - mean[c]).abs() < 5.0 * sd[c] / count[st].sqrt());
        }
    }
    let cat = [
        Emitter::Categorical {
            probabilities: vec![0.9, 0.1, 0.0],
        },
        Emitter::Categorical {
            probabilities: vec![0.0, 0.2, 0.8],
        },
    ];
    let s = hmm_sample(&[0.5, 0.5], &a, &cat, &[1000], 1).unwrap();
    for (st, o) in s.states[0].iter().zip(&s.observations[0]) {
        assert!(if *st == 0 { *o < 2.0 } else { *o > 0.0 });
    }
}

#[test]
fn layered_field_and_drillholes() {
    let grid = Grid::new([0.0, 0.0, -50.0], [10.0, 10.0, 0.5], [5, 4, 100]).unwrap();
    let q = RateMatrix::new(golden_q()).unwrap();
    let field = layered_field(&grid, &q, 2).unwrap();
    assert_eq!(field.len(), grid.n_cells());
    for i in 0..grid.n_cells() {
        let [_, _, k] = grid.ijk(i);
        assert_eq!(field[i], field[grid.index([0, 0, k])]);
    }
    let holes = drillholes(&grid, &field, &[[0, 0], [4, 3]]).unwrap();
    assert_eq!(holes.codes.len(), 200);
    assert_eq!(holes.from[0], 0.0);
    assert_eq!(holes.to[99], 50.0);
    assert_eq!(holes.codes[0], field[grid.index([0, 0, 99])]);
    let seqs =
        SequenceSet::from_intervals(&holes.holes, &holes.from, &holes.to, &holes.codes, 3).unwrap();
    assert_eq!(seqs.runs(Direction::Up, 1e-9).len(), 2);
    assert!(drillholes(&grid, &field, &[[5, 0]]).is_err());
}

fn run_in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn threads_do_not_change_results() {
    let p = TransitionMatrix::new(expm_rate(&golden_q(), 1.0)).unwrap();
    let lengths: Vec<usize> = (0..40).map(|i| 100 + 7 * i).collect();
    let one = run_in_pool(1, || markov_holes(&p, &lengths, 5).unwrap());
    let four = run_in_pool(4, || markov_holes(&p, &lengths, 5).unwrap());
    assert_eq!(one, four);
    let q = RateMatrix::new(golden_q()).unwrap();
    let ls = vec![80.0; 40];
    let one = run_in_pool(1, || rate_holes(&q, &ls, Direction::Down, 5).unwrap());
    let four = run_in_pool(4, || rate_holes(&q, &ls, Direction::Down, 5).unwrap());
    assert_eq!(one, four);
    let em = [
        Emitter::Gaussian {
            mean: vec![0.0],
            sd: vec![1.0],
        },
        Emitter::Gaussian {
            mean: vec![1.0],
            sd: vec![1.0],
        },
        Emitter::Gaussian {
            mean: vec![2.0],
            sd: vec![1.0],
        },
    ];
    let one = run_in_pool(1, || {
        hmm_sample(&[0.2, 0.3, 0.5], &p, &em, &lengths, 5).unwrap()
    });
    let four = run_in_pool(4, || {
        hmm_sample(&[0.2, 0.3, 0.5], &p, &em, &lengths, 5).unwrap()
    });
    assert_eq!(
        one.observations
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        four.observations
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

fn stochastic(k: usize) -> impl Strategy<Value = Mat> {
    proptest::collection::vec(0.01f64..1.0, k * k).prop_map(move |v| {
        let mut m = Mat::from_row_slice(k, k, &v);
        for mut row in m.row_iter_mut() {
            let s: f64 = row.iter().sum();
            row /= s;
        }
        m
    })
}

fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
    v: &T,
) {
    let text = serde_json::to_string(v).unwrap();
    assert_eq!(&serde_json::from_str::<T>(&text).unwrap(), v, "{text}");
}

proptest! {
    #[test]
    fn serde_round_trips(
        p in (1usize..6).prop_flat_map(stochastic),
        origin in proptest::array::uniform3(-1e6f64..1e6),
        size in proptest::array::uniform3(0.01f64..100.0),
        count in proptest::array::uniform3(1usize..50),
        names in proptest::collection::hash_set("[a-z]{1,8}", 1..6),
        distance in 0.0f64..1.0,
    ) {
        let t = TransitionMatrix::new(p.clone()).unwrap();
        round_trip(&t);
        let k = p.nrows();
        let q = Mat::from_fn(k, k, |i, j| if i == j { 0.0 } else { p[(i, j)] });
        let mut q = q;
        for i in 0..k {
            let s: f64 = q.row(i).sum();
            q[(i, i)] = -s;
        }
        if k > 1 {
            round_trip(&RateMatrix::new(q).unwrap());
        }
        round_trip(&Grid::new(origin, size, count).unwrap());
        let names: Vec<String> = names.into_iter().collect();
        let colors = Some(names.iter().map(|n| format!("#{n}")).collect());
        round_trip(&Categories::new(names.clone(), None).unwrap());
        round_trip(&Categories::new(names, colors).unwrap());
        round_trip(&Projection { distance, exact: distance == 0.0 });
        round_trip(&Direction::Up);
        round_trip(&EmptyRow::Uniform);
    }

    #[test]
    fn transition_rows_stay_stochastic(p in (2usize..6).prop_flat_map(stochastic), h in 0.0f64..50.0) {
        let k = p.nrows();
        let mut q = Mat::from_fn(k, k, |i, j| if i == j { 0.0 } else { p[(i, j)] });
        for i in 0..k {
            let s: f64 = q.row(i).sum();
            q[(i, i)] = -s;
        }
        let t = RateMatrix::new(q).unwrap().transition(h);
        for i in 0..k {
            prop_assert!((t.matrix().row(i).sum() - 1.0).abs() < 1e-12);
        }
        prop_assert!(t.matrix().iter().all(|&v| v >= 0.0));
    }
}
