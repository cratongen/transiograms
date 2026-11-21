use std::process::Command;

use proptest::prelude::*;
use transiograms_chain::*;
use transiograms_core::rng::tag;
use transiograms_core::special::chi2_sf;
use transiograms_core::synthetic::{markov_holes, rate_holes};
use transiograms_core::{
    Code, Direction, EmptyRow, MISSING, Mat, RateMatrix, SequenceSet, Stream, TransitionMatrix,
};

fn golden_q() -> RateMatrix {
    RateMatrix::new(Mat::from_row_slice(
        3,
        3,
        &[-0.25, 0.15, 0.10, 0.35, -0.5, 0.15, 0.1, 0.375, -0.475],
    ))
    .unwrap()
}

fn golden_counts() -> Mat {
    Mat::from_row_slice(
        4,
        4,
        &[
            0., 30., 10., 5., 8., 0., 25., 12., 20., 6., 0., 9., 10., 14., 8., 0.,
        ],
    )
}

fn chain_with_counts(n: Mat) -> MarkovChain {
    let p = TransitionMatrix::from_counts(&n, 0.0, true, EmptyRow::Raise).unwrap();
    let mut c = MarkovChain::from_transition(p, Kind::Embedded, None, Direction::Up).unwrap();
    c.counts = Some(n);
    c
}

fn fixed_opts(step: f64) -> FitOptions {
    FitOptions {
        kind: Kind::Fixed { step },
        direction: Direction::Down,
        ..Default::default()
    }
}

fn embedded_opts() -> FitOptions {
    FitOptions {
        direction: Direction::Down,
        ..Default::default()
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Asymptotic Kolmogorov p value with the Stephens small-sample correction.
fn ks_uniform_p(mut p: Vec<f64>) -> f64 {
    p.sort_by(f64::total_cmp);
    let n = p.len() as f64;
    let d = p
        .iter()
        .enumerate()
        .map(|(i, &v)| ((i + 1) as f64 / n - v).max(v - i as f64 / n))
        .fold(0.0, f64::max);
    let lambda = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
    let s: f64 = (1..200)
        .map(|k| {
            let k = k as f64;
            2.0 * (-1f64).powf(k - 1.0) * (-2.0 * k * k * lambda * lambda).exp()
        })
        .sum();
    s.clamp(0.0, 1.0)
}

fn check_chain(c: &MarkovChain, zero_diagonal: bool) {
    let p = c.transition.matrix();
    for i in 0..c.k() {
        assert!((p.row(i).sum() - 1.0).abs() <= 1e-12);
    }
    if zero_diagonal {
        assert!((0..c.k()).all(|i| p[(i, i)] == 0.0));
    }
    if let Some(q) = &c.generator {
        let q = q.matrix();
        let scale = q.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for i in 0..c.k() {
            assert!(q.row(i).sum().abs() <= 1e-12 * scale);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn fitted_chains_are_stochastic(
        holes in prop::collection::vec(prop::collection::vec(0u32..4, 2..40), 1..5),
        a in prop_oneof![Just(0.0), 0.01f64..2.0],
        uniform in any::<bool>(),
    ) {
        let mut holes = holes;
        holes[0].splice(0..0, [0, 1, 2, 3]);
        let seqs = SequenceSet::from_regular(&holes, 4).unwrap();
        let on_empty_row = if uniform { EmptyRow::Uniform } else { EmptyRow::Raise };
        for kind in [Kind::Embedded, Kind::Fixed { step: 1.0 }] {
            for self_transitions in [false, true] {
                let opts = FitOptions { kind, self_transitions, pseudocount: a, on_empty_row, ..Default::default() };
                if let Ok(c) = MarkovChain::fit(&seqs, &opts) {
                    check_chain(&c, kind == Kind::Embedded && !self_transitions);
                }
            }
        }
        let opts = FitOptions { pseudocount: a.max(0.1), ..Default::default() };
        check_chain(&MarkovChain::fit(&seqs, &opts).unwrap(), true);
    }
}

#[test]
fn built_chains_are_stochastic() {
    let q = golden_q();
    check_chain(
        &MarkovChain::from_rates(q.clone(), Direction::Up).unwrap(),
        true,
    );
    check_chain(
        &MarkovChain::from_transition(
            q.transition(0.5),
            Kind::Fixed { step: 0.5 },
            None,
            Direction::Up,
        )
        .unwrap(),
        false,
    );
    check_chain(
        &MarkovChain::from_geology(
            &[0.5, 0.3, 0.2],
            &[4.0, 2.0, 3.0],
            None,
            2,
            false,
            Direction::Up,
        )
        .unwrap(),
        true,
    );
}

#[test]
fn independence_golden_4x4() {
    let c = chain_with_counts(golden_counts());
    let x2 = c.independence_test(Method::Pearson).unwrap();
    let g = c.independence_test(Method::G).unwrap();
    assert!(rel(x2.statistic, 29.298020503608846) < 1e-9);
    assert!(rel(g.statistic, 31.237967935478377) < 1e-9);
    assert_eq!((x2.df, g.df), (5, 5));
    assert!(rel(x2.p_value, 2.0266e-05) < 1e-4);
    assert!(rel(g.p_value, 8.4064e-06) < 1e-4);
    let e = g.expected.unwrap();
    let golden = [
        [0.0, 20.6805682797, 15.4762764461, 8.8431552742],
        [17.022383202, 0.0, 17.8042536869, 10.1733631111],
        [11.68497564, 16.3315427453, 0.0, 6.9834816147],
        [9.292641158, 12.987888975, 9.719469867, 0.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            assert!((e[(i, j)] - golden[i][j]).abs() < 1e-8);
        }
    }
    let gingerich = [0.0, 18.080, 14.063, 12.857];
    assert!((1..4).all(|j| (e[(0, j)] - gingerich[j]).abs() > 0.1));
    let d = c.difference_matrix().unwrap();
    for i in 0..4 {
        assert!(d.row(i).sum().abs() < 1e-12);
    }
}

#[test]
fn independence_golden_3x3_cycle() {
    let n = Mat::from_row_slice(3, 3, &[0., 12., 7., 5., 0., 14., 9., 4., 0.]);
    let e = chain_with_counts(n)
        .independence_test(Method::G)
        .unwrap()
        .expected
        .unwrap();
    let golden = [
        [0.0, 8.919987347, 10.080012653],
        [8.080012653, 0.0, 10.919987347],
        [5.919987347, 7.080012653, 0.0],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert!((e[(i, j)] - golden[i][j]).abs() < 1e-8);
        }
    }
    let lhs = e[(0, 1)] * e[(1, 2)] * e[(2, 0)];
    let rhs = e[(0, 2)] * e[(2, 1)] * e[(1, 0)];
    assert!(rel(lhs, rhs) < 1e-10);
}

#[test]
fn two_category_embedded_raises() {
    let c = chain_with_counts(Mat::from_row_slice(2, 2, &[0., 7., 7., 0.]));
    assert!(c.independence_test(Method::G).is_err());
}

#[test]
fn never_tallies_across_holes_gaps_or_missing() {
    let holes: Vec<String> = ["a", "a", "a", "a", "a", "b", "b"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let from = [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0];
    let to = [1.0, 2.0, 3.0, 4.0, 6.0, 1.0, 2.0];
    let codes = [0, 1, MISSING, 2, 0, 2, 1];
    let seqs = SequenceSet::from_intervals(&holes, &from, &to, &codes, 3).unwrap();
    let opts = FitOptions {
        on_empty_row: EmptyRow::Uniform,
        ..embedded_opts()
    };
    let n = MarkovChain::fit(&seqs, &opts).unwrap().counts.unwrap();
    let mut want = Mat::zeros(3, 3);
    want[(0, 1)] = 1.0;
    want[(2, 1)] = 1.0;
    assert_eq!(n, want);
    let f = MarkovChain::fit(
        &seqs,
        &FitOptions {
            kind: Kind::Fixed { step: 1.0 },
            ..opts
        },
    )
    .unwrap()
    .counts
    .unwrap();
    assert_eq!(f, want);
}

#[test]
fn proportion_std_golden() {
    let c = MarkovChain::from_transition(
        golden_q().transition(0.5),
        Kind::Fixed { step: 0.5 },
        None,
        Direction::Down,
    )
    .unwrap();
    let sd = c.proportion_std(100_000).unwrap();
    for (a, b) in sd.iter().zip([0.00448, 0.00346, 0.00335]) {
        assert!((a - b).abs() < 5e-6, "{sd:?}");
    }
}

fn samples(codes: &[Vec<Code>], step: f64, k: usize) -> SequenceSet {
    let mut holes = vec![];
    let mut depth = vec![];
    let mut all = vec![];
    for (h, s) in codes.iter().enumerate() {
        for (i, &c) in s.iter().enumerate() {
            holes.push(h.to_string());
            depth.push((i as f64 + 0.5) * step);
            all.push(c);
        }
    }
    SequenceSet::from_samples(&holes, &depth, &all, step, k).unwrap()
}

#[test]
fn m1_acceptance_fixed_recovery() {
    let truth = MarkovChain::from_transition(
        golden_q().transition(0.5),
        Kind::Fixed { step: 0.5 },
        None,
        Direction::Down,
    )
    .unwrap();
    let Simulated::Codes(codes) = truth
        .simulate(
            Target::Steps(100_000),
            &Hard::default(),
            Start::Stationary,
            1,
            0,
        )
        .unwrap()
    else {
        panic!()
    };
    let fit = MarkovChain::fit(&samples(&codes, 0.5, 3), &fixed_opts(0.5)).unwrap();
    let c = fit.counts.as_ref().unwrap();
    let (p, ph) = (truth.transition.matrix(), fit.transition.matrix());
    for i in 0..3 {
        let ci: f64 = c.row(i).sum();
        for j in 0..3 {
            let tol = 3.0 * (p[(i, j)] * (1.0 - p[(i, j)]) / ci).sqrt();
            assert!((ph[(i, j)] - p[(i, j)]).abs() <= tol, "P[{i},{j}]");
        }
    }
    let sd = truth.proportion_std(100_000).unwrap();
    for k in 0..3 {
        assert!(
            (fit.proportions[k] - truth.proportions[k]).abs() <= 3.0 * sd[k],
            "p[{k}]"
        );
    }
    let q = fit.generator.as_ref().unwrap();
    assert!(fit.projection.unwrap().exact);
    let back = q.transition(0.5);
    assert!((back.matrix() - ph).abs().max() < 1e-10);
}

#[test]
fn generator_of_exact_transition_is_recovered() {
    let q = golden_q();
    for h in [0.5, 1.0, 5.0] {
        let c = MarkovChain::from_transition(
            q.transition(h),
            Kind::Fixed { step: h },
            None,
            Direction::Up,
        )
        .unwrap();
        let g = c.generator.as_ref().unwrap();
        assert!((g.matrix() - q.matrix()).abs().max() < 1e-9);
        assert!(
            (g.transition(h).matrix() - c.transition.matrix())
                .abs()
                .max()
                < 1e-12
        );
        let p3 = c.transition_at(3.0 * h).unwrap();
        assert!((p3.matrix() - q.transition(3.0 * h).matrix()).abs().max() < 1e-12);
    }
}

#[test]
fn embedded_mle_recovers_lengths_and_rates() {
    let q = golden_q();
    let iv = rate_holes(&q, &vec![60.0; 400], Direction::Down, 3).unwrap();
    let seqs = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
    let c = MarkovChain::fit(&seqs, &embedded_opts()).unwrap();
    let l = q.mean_lengths();
    let lh = c.mean_lengths.as_ref().unwrap();
    let n = c.counts.as_ref().unwrap();
    let t = c.exposure.as_ref().unwrap();
    for j in 0..3 {
        let exits: f64 = n.row(j).sum();
        assert!((lh[j] - l[j]).abs() <= 3.0 * l[j] / exits.sqrt(), "L[{j}]");
    }
    let qh = c.generator.as_ref().unwrap().matrix();
    for j in 0..3 {
        for k in 0..3 {
            let sd = (q.matrix()[(j, k)].abs() / t[j]).sqrt();
            assert!(
                (qh[(j, k)] - q.matrix()[(j, k)]).abs() <= 3.0 * sd,
                "Q[{j},{k}]"
            );
        }
    }
    let nu = &c.bed_frequencies;
    let s: f64 = (0..3).map(|j| nu[j] * lh[j]).sum();
    for k in 0..3 {
        assert!((c.proportions[k] - nu[k] * lh[k] / s).abs() < 1e-12);
    }
}

fn indep_seqs(n: usize, len: usize, seed: u64) -> Vec<SequenceSet> {
    (0..n)
        .map(|r| {
            let mut st = Stream::new(seed, &[tag::SAMPLE, r as u64]);
            let s: Vec<Code> = (0..len)
                .map(|_| st.categorical(&[0.5, 0.3, 0.2]) as Code)
                .collect();
            SequenceSet::from_regular(&[s], 3).unwrap()
        })
        .collect()
}

#[test]
fn order_zero_sequences_reject_at_nominal_rate() {
    let p: Vec<f64> = indep_seqs(1000, 1000, 11)
        .iter()
        .map(|s| {
            order_test(s, &fixed_opts(1.0), 1, Method::G)
                .unwrap()
                .p_value
        })
        .collect();
    assert!(ks_uniform_p(p.clone()) > 0.001);
    let rejected = p.iter().filter(|&&v| v < 0.05).count() as f64;
    assert!(
        (rejected - 50.0).abs() <= 3.0 * (1000.0f64 * 0.05 * 0.95).sqrt(),
        "{rejected}"
    );
    let t = order_test(
        &indep_seqs(1, 2000, 11)[0],
        &fixed_opts(1.0),
        1,
        Method::Pearson,
    )
    .unwrap();
    assert_eq!(t.df, 4);
}

#[test]
fn first_order_chain_gives_uniform_order2_p_values() {
    let p = golden_q().transition(0.5);
    let holes = markov_holes(&p, &[2000; 200], 5).unwrap();
    let pv: Vec<f64> = holes
        .iter()
        .map(|h| {
            let s = SequenceSet::from_regular(std::slice::from_ref(h), 3).unwrap();
            let t = order_test(&s, &fixed_opts(1.0), 2, Method::G).unwrap();
            assert_eq!(t.df, 12);
            t.p_value
        })
        .collect();
    assert!(ks_uniform_p(pv) > 0.001);
    let s = SequenceSet::from_regular(&holes[..20], 3).unwrap();
    assert_eq!(
        order_test(&s, &fixed_opts(1.0), 3, Method::G).unwrap().df,
        36
    );
    let t = select_order(&s, &fixed_opts(1.0), 3).unwrap();
    assert_eq!(t.n_transitions, 20 * (2000 - 3));
    assert!(t.log_likelihood.windows(2).all(|w| w[1] >= w[0] - 1e-9));
    let best = (0..4)
        .min_by(|&a, &b| t.bic[a].total_cmp(&t.bic[b]))
        .unwrap();
    assert_eq!(best, 1);
}

#[test]
fn second_order_chain_is_detected() {
    let mut st = Stream::new(9, &[]);
    let mut s: Vec<Code> = vec![0, 1];
    for t in 2..3000 {
        let next = if st.uniform() < 0.6 {
            s[t - 2]
        } else {
            st.below(3) as Code
        };
        s.push(next);
    }
    let seqs = SequenceSet::from_regular(&[s], 3).unwrap();
    assert!(
        order_test(&seqs, &fixed_opts(1.0), 2, Method::G)
            .unwrap()
            .p_value
            < 1e-6
    );
    let t = select_order(&seqs, &fixed_opts(1.0), 3).unwrap();
    let best = (0..4)
        .min_by(|&a, &b| t.bic[a].total_cmp(&t.bic[b]))
        .unwrap();
    assert_eq!(best, 2);
}

#[test]
fn embedded_order2_test_on_first_order_chain() {
    let pi = golden_q().embedded();
    let holes = markov_holes(&pi, &[5000], 2).unwrap();
    let seqs = SequenceSet::from_regular(&holes, 3).unwrap();
    let t = order_test(&seqs, &embedded_opts(), 2, Method::G).unwrap();
    assert_eq!(t.df, 3);
    assert!(t.p_value > 0.001);
    assert!(order_test(&seqs, &embedded_opts(), 3, Method::G).is_err());
}

#[test]
fn homogeneity_of_one_chain() {
    let p = golden_q().transition(0.5);
    let holes = markov_holes(&p, &[3000, 3000], 8).unwrap();
    let groups: Vec<SequenceSet> = holes
        .iter()
        .map(|h| SequenceSet::from_regular(std::slice::from_ref(h), 3).unwrap())
        .collect();
    let t = homogeneity_test(&groups, &fixed_opts(1.0), Method::G).unwrap();
    assert_eq!(t.df, 6);
    assert!(t.p_value > 0.001);
    let tp = homogeneity_test(&groups, &fixed_opts(1.0), Method::Pearson).unwrap();
    assert!(tp.p_value > 0.001);
    let te = homogeneity_test(&groups, &embedded_opts(), Method::G).unwrap();
    assert_eq!(te.df, 3);
}

#[test]
fn bowker_symmetric_is_zero() {
    let n = Mat::from_row_slice(3, 3, &[0., 4., 6., 4., 0., 9., 6., 9., 0.]);
    let t = chain_with_counts(n).symmetry_test().unwrap();
    assert_eq!(t.statistic, 0.0);
    assert_eq!(t.df, 3);
    assert_eq!(t.p_value, 1.0);
}

#[test]
fn entropy_of_uniform_embedded_rows() {
    let m = 4;
    let pi = Mat::from_fn(m, m, |i, j| if i == j { 0.0 } else { 1.0 / 3.0 });
    let c = MarkovChain::from_transition(
        TransitionMatrix::new(pi).unwrap(),
        Kind::Embedded,
        None,
        Direction::Up,
    )
    .unwrap();
    let e = c.entropy().unwrap();
    for i in 0..m {
        assert!((e.post[i] - 3f64.log2()).abs() < 1e-12);
        assert!((e.post_normalized[i] - 1.0).abs() < 1e-12);
        assert!((e.pre[i] - 3f64.log2()).abs() < 1e-12);
    }
    assert!((e.max - 3f64.log2()).abs() < 1e-15);
    assert!(c.asymmetry().unwrap().abs().max() < 1e-15);
}

fn binom_close(count: usize, n: usize, p: f64) -> bool {
    let n = n as f64;
    (count as f64 - n * p).abs() <= 3.0 * (n * p * (1.0 - p)).sqrt() + 1e-9
}

#[test]
fn discrete_bridge_monte_carlo() {
    let pm = golden_q().transition(0.5);
    let chain =
        MarkovChain::from_transition(pm.clone(), Kind::Fixed { step: 0.5 }, None, Direction::Up)
            .unwrap();
    let (a, b, n) = (0usize, 2usize, 6usize);
    let hard = Hard {
        from: vec![0.0, 6.0],
        to: vec![0.0, 6.0],
        codes: vec![a as Code, b as Code],
    };
    let reps = 100_000;
    let Simulated::Codes(paths) = chain
        .simulate(Target::Steps(n + 1), &hard, Start::Stationary, reps, 1)
        .unwrap()
    else {
        panic!()
    };
    let p = pm.matrix();
    let pn = pm.power(n as u64)[(a, b)];
    let mut freq = vec![0usize; 243];
    for path in &paths {
        assert_eq!((path[0], path[n]), (a as Code, b as Code));
        freq[path[1..n].iter().fold(0, |s, &c| s * 3 + c as usize)] += 1;
    }
    let (mut stat, mut cells, mut pool_o, mut pool_e) = (0.0, 0, 0.0, 0.0);
    for (idx, &f) in freq.iter().enumerate() {
        let mut x = vec![a];
        x.extend((0..5).rev().map(|d| idx / 3usize.pow(d) % 3));
        x.push(b);
        let pr: f64 = x.windows(2).map(|w| p[(w[0], w[1])]).product::<f64>() / pn;
        let e = pr * reps as f64;
        if e < 5.0 {
            pool_o += f as f64;
            pool_e += e;
        } else {
            stat += (f as f64 - e).powi(2) / e;
            cells += 1;
        }
    }
    if pool_e > 0.0 {
        stat += (pool_o - pool_e).powi(2) / pool_e;
        cells += 1;
    }
    assert!(
        chi2_sf(stat, (cells - 1) as f64) > 0.001,
        "{stat} on {cells}"
    );
    for t in 1..n {
        let pt = pm.power(t as u64);
        let pr = pm.power((n - t) as u64);
        for j in 0..3 {
            let want = pt[(a, j)] * pr[(j, b)] / pn;
            let got = paths.iter().filter(|x| x[t] as usize == j).count();
            assert!(binom_close(got, reps, want), "t {t} j {j}");
        }
    }
}

#[test]
fn discrete_prefix_uses_reversed_chain() {
    let pm = golden_q().transition(0.5);
    let chain =
        MarkovChain::from_transition(pm.clone(), Kind::Fixed { step: 0.5 }, None, Direction::Up)
            .unwrap();
    let hard = Hard {
        from: vec![5.0],
        to: vec![5.0],
        codes: vec![1],
    };
    let reps = 50_000;
    let Simulated::Codes(paths) = chain
        .simulate(Target::Steps(10), &hard, Start::Stationary, reps, 2)
        .unwrap()
    else {
        panic!()
    };
    let pi = &chain.proportions;
    let p5 = pm.power(5);
    for j in 0..3 {
        let want = pi[j] * p5[(j, 1)] / pi[1];
        let got = paths.iter().filter(|x| x[0] as usize == j).count();
        assert!(binom_close(got, reps, want));
        let after = paths.iter().filter(|x| x[6] as usize == j).count();
        assert!(binom_close(after, reps, pm.matrix()[(1, j)]));
    }
}

#[test]
fn impossible_hard_data_raise() {
    let c = MarkovChain::from_rates(golden_q(), Direction::Up).unwrap();
    let hard = Hard {
        from: vec![2.0, 3.0],
        to: vec![2.0, 3.0],
        codes: vec![0, 0],
    };
    assert!(
        c.simulate(Target::Beds(6), &hard, Start::Stationary, 1, 0)
            .is_err()
    );
    let span = Hard {
        from: vec![2.0],
        to: vec![3.0],
        codes: vec![1],
    };
    assert!(
        c.simulate(Target::Beds(6), &span, Start::Stationary, 1, 0)
            .is_err()
    );
    let ok = Hard {
        from: vec![2.0, 3.0],
        to: vec![2.0, 3.0],
        codes: vec![0, 1],
    };
    let Simulated::Codes(beds) = c
        .simulate(Target::Beds(6), &ok, Start::Stationary, 50, 0)
        .unwrap()
    else {
        panic!()
    };
    for s in beds {
        assert_eq!((s[2], s[3]), (0, 1));
        assert!(s.windows(2).all(|w| w[0] != w[1]));
    }
}

fn code_at(beds: &(Vec<f64>, Vec<f64>, Vec<Code>), x: f64) -> usize {
    let i = beds.0.iter().rposition(|&a| a <= x).unwrap();
    assert!(x <= beds.1[i]);
    beds.2[i] as usize
}

#[test]
fn continuous_bridge_marginals_and_hard_data() {
    let q = golden_q();
    let c = MarkovChain::from_rates(q.clone(), Direction::Up).unwrap();
    let hard = Hard {
        from: vec![1.0, 4.0],
        to: vec![2.0, 5.0],
        codes: vec![0, 2],
    };
    let reps = 40_000;
    let Simulated::Beds(out) = c
        .simulate(Target::Length(6.0), &hard, Start::Stationary, reps, 4)
        .unwrap()
    else {
        panic!()
    };
    for r in &out {
        assert_eq!(r.0[0], 0.0);
        assert_eq!(*r.1.last().unwrap(), 6.0);
        assert!(r.0.iter().skip(1).zip(&r.1).all(|(a, b)| a == b));
        assert!(r.2.windows(2).all(|w| w[0] != w[1]));
        for x in [1.0, 1.5, 1.999] {
            assert_eq!(code_at(r, x), 0);
        }
        for x in [4.0, 4.5, 4.999] {
            assert_eq!(code_at(r, x), 2);
        }
    }
    let p = q.stationary().unwrap();
    let (t1, t2) = (q.transition(1.0), q.transition(2.0));
    let half = q.transition(0.5);
    for j in 0..3 {
        let mid = t1.matrix()[(0, j)] * t1.matrix()[(j, 2)] / t2.matrix()[(0, 2)];
        let got = out.iter().filter(|r| code_at(r, 3.0) == j).count();
        assert!(binom_close(got, reps, mid), "bridge j {j}");
        let before = p[j] * t1.matrix()[(j, 0)] / p[0];
        let got = out.iter().filter(|r| code_at(r, 0.0) == j).count();
        assert!(binom_close(got, reps, before), "prefix j {j}");
        let got = out.iter().filter(|r| code_at(r, 5.5) == j).count();
        assert!(
            binom_close(got, reps, half.matrix()[(2, j)]),
            "suffix j {j}"
        );
    }
}

#[test]
fn refit_of_simulated_beds_recovers_q() {
    let q = golden_q();
    let c = MarkovChain::from_rates(q.clone(), Direction::Down).unwrap();
    let Simulated::Beds(out) = c
        .simulate(
            Target::Length(100.0),
            &Hard::default(),
            Start::Stationary,
            300,
            6,
        )
        .unwrap()
    else {
        panic!()
    };
    let (mut holes, mut from, mut to, mut codes) = (vec![], vec![], vec![], vec![]);
    for (h, r) in out.iter().enumerate() {
        for i in 0..r.2.len() {
            holes.push(h.to_string());
            from.push(r.0[i]);
            to.push(r.1[i]);
            codes.push(r.2[i]);
        }
    }
    let seqs = SequenceSet::from_intervals(&holes, &from, &to, &codes, 3).unwrap();
    let fit = MarkovChain::fit(&seqs, &embedded_opts()).unwrap();
    let t = fit.exposure.as_ref().unwrap();
    let qh = fit.generator.as_ref().unwrap().matrix();
    for j in 0..3 {
        for k in 0..3 {
            let sd = (q.matrix()[(j, k)].abs() / t[j]).sqrt();
            assert!(
                (qh[(j, k)] - q.matrix()[(j, k)]).abs() <= 3.0 * sd,
                "Q[{j},{k}]"
            );
        }
    }
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let c = MarkovChain::from_rates(golden_q(), Direction::Up).unwrap();
    let f = MarkovChain::from_transition(
        golden_q().transition(0.5),
        Kind::Fixed { step: 0.5 },
        None,
        Direction::Up,
    )
    .unwrap();
    let hard = Hard {
        from: vec![3.0],
        to: vec![4.0],
        codes: vec![1],
    };
    let run = || {
        let a = c
            .simulate(Target::Length(20.0), &hard, Start::Stationary, 64, 7)
            .unwrap();
        let b = f
            .simulate(Target::Steps(40), &hard, Start::Stationary, 64, 7)
            .unwrap();
        let iv = rate_holes(&golden_q(), &[200.0; 4], Direction::Up, 1).unwrap();
        let s = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
        let e = sojourn(&s, Direction::Up, 1e-6)
            .exponential_test(0, 199, 3)
            .unwrap();
        (a, b, e)
    };
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
    };
    assert_eq!(pool(1).install(run), pool(4).install(run));
}

#[test]
fn geology_golden_carle() {
    let mut pi = Mat::zeros(3, 3);
    pi[(0, 1)] = 0.6;
    pi[(1, 0)] = 0.7;
    let c = MarkovChain::from_geology(
        &[0.5, 0.3, 0.2],
        &[4.0, 2.0, 1.0],
        Some(&pi),
        2,
        false,
        Direction::Up,
    )
    .unwrap();
    let want = golden_q();
    assert!(
        (c.generator.as_ref().unwrap().matrix() - want.matrix())
            .abs()
            .max()
            < 1e-12
    );
    let l = c.mean_lengths.as_ref().unwrap();
    assert!((l[2] - 2.105263157894737).abs() < 1e-12);
    for (a, b) in c
        .bed_frequencies
        .iter()
        .zip([0.3378378378, 0.4054054054, 0.2567567568])
    {
        assert!((a - b).abs() < 1e-9);
    }
    let sym = MarkovChain::from_geology(
        &[0.4, 0.4, 0.2],
        &[2.0, 2.0, 1.0],
        Some(&Mat::from_element(3, 3, 0.5)),
        2,
        true,
        Direction::Up,
    )
    .unwrap();
    let q = sym.generator.unwrap();
    assert!(q.is_reversible(&[0.4, 0.4, 0.2], 1e-12));
}

#[test]
fn sojourn_survival_and_exponential_test() {
    let iv = rate_holes(&golden_q(), &[100.0; 100], Direction::Down, 12).unwrap();
    let s = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
    let so = sojourn(&s, Direction::Down, 1e-6);
    let l = golden_q().mean_lengths();
    for c in 0..3 {
        let closed = so.open[c].iter().filter(|o| !**o).count();
        assert!(so.n_open[c] < so.n_beds[c]);
        assert!((so.mean_length[c] - l[c]).abs() <= 3.0 * l[c] / (closed as f64).sqrt());
        let (t, sv) = so.survival(c as Code);
        assert!(t.windows(2).all(|w| w[0] < w[1]));
        assert!(sv.windows(2).all(|w| w[1] <= w[0]));
        let p = so.exponential_test(c as Code, 199, 0).unwrap().p_value;
        assert!(p > 0.001, "{p}");
    }
    let uniform = Sojourn {
        n_beds: vec![4],
        n_open: vec![1],
        total: vec![10.0],
        mean_length: vec![10.0 / 3.0],
        thickness: vec![vec![1.0, 2.0, 3.0, 4.0]],
        open: vec![vec![false, false, true, false]],
    };
    let (t, sv) = uniform.survival(0);
    assert_eq!(t, vec![1.0, 2.0, 4.0]);
    let want = [0.75, 0.5, 0.0];
    assert!(sv.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-15));
    let flat = Sojourn {
        thickness: vec![vec![1.0; 50]],
        open: vec![vec![false; 50]],
        ..uniform
    };
    assert!(flat.exponential_test(0, 199, 0).unwrap().p_value < 0.01);
}

#[test]
fn serde_round_trip() {
    let iv = rate_holes(&golden_q(), &[50.0; 20], Direction::Up, 1).unwrap();
    let s = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
    let c = MarkovChain::fit(&s, &FitOptions::default()).unwrap();
    let back: MarkovChain = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back, c);
    let t = c.independence_test(Method::G).unwrap();
    let back: TestResult = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t);
}

fn no_repeat_sequence() -> SequenceSet {
    let pattern: [Code; 9] = [0, 1, 2, 0, 2, 1, 0, 1, 2];
    let s: Vec<Code> = pattern.iter().copied().cycle().take(900).collect();
    SequenceSet::from_regular(&[s], 3).unwrap()
}

/// Reruns the ignored test `name` in a child capped at `kb` KiB of address space, one thread,
/// 300 s; an allocation the library should have refused aborts the child instead of the machine.
fn bounded(name: &str, kb: u64) -> std::process::ExitStatus {
    let exe = std::env::current_exe().unwrap();
    Command::new("sh")
        .arg("-c")
        .arg(format!(
            "ulimit -v {kb}; exec timeout 300 \"$0\" --exact {name} --ignored --test-threads 1"
        ))
        .arg(exe)
        .env("RAYON_NUM_THREADS", "1")
        .env("MALLOC_ARENA_MAX", "1")
        .status()
        .unwrap()
}

fn assert_bounded(name: &str, kb: u64) {
    let s = bounded(name, kb);
    assert!(
        s.success(),
        "{name} failed under a {kb} KiB address-space cap: {s:?}"
    );
}

#[test]
fn self_transition_independence_uses_ordinary_contingency() {
    let seqs = no_repeat_sequence();
    let opts = FitOptions {
        self_transitions: true,
        ..Default::default()
    };
    let c = MarkovChain::fit(&seqs, &opts).unwrap();
    let n = c.counts.clone().unwrap();
    let rows: Vec<f64> = (0..3).map(|i| n.row(i).sum()).collect();
    let cols: Vec<f64> = (0..3).map(|j| n.column(j).sum()).collect();
    let total: f64 = rows.iter().sum();
    let t = c.independence_test(Method::Pearson).unwrap();
    assert_eq!(
        t.df, 4,
        "chain.md section 3: self_transitions gives df (m'-1)^2"
    );
    let e = t.expected.unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let want = rows[i] * cols[j] / total;
            assert!(
                (e[(i, j)] - want).abs() < 1e-9,
                "E[{i},{j}] = {} but r_i c_j / N = {want}",
                e[(i, j)]
            );
        }
    }
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_order_test_huge_order() {
    let s: Vec<Code> = (0..40).map(|i| (i % 4) as Code).collect();
    let seqs = SequenceSet::from_regular(&[s], 4).unwrap();
    let opts = FitOptions {
        kind: Kind::Fixed { step: 1.0 },
        ..Default::default()
    };
    let r = std::panic::catch_unwind(|| order_test(&seqs, &opts, 33, Method::G));
    assert!(matches!(r, Ok(Err(_))), "order 33 with m = 4 panicked");
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn order_test_with_huge_order_raises_instead_of_panicking() {
    assert_bounded("inner_order_test_huge_order", 2_000_000);
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_select_order_huge_order() {
    let s: Vec<Code> = (0..40).map(|i| (i % 4) as Code).collect();
    let seqs = SequenceSet::from_regular(&[s], 4).unwrap();
    let opts = FitOptions {
        kind: Kind::Fixed { step: 1.0 },
        ..Default::default()
    };
    let r = std::panic::catch_unwind(|| select_order(&seqs, &opts, 33));
    assert!(matches!(r, Ok(Err(_))), "max_order 33 with m = 4 panicked");
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn select_order_with_huge_order_raises_instead_of_panicking() {
    assert_bounded("inner_select_order_huge_order", 2_000_000);
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_order_test_order_beyond_the_data() {
    let s: Vec<Code> = (0..2000).map(|i| (i % 4) as Code).collect();
    let seqs = SequenceSet::from_regular(&[s], 4).unwrap();
    let opts = FitOptions {
        kind: Kind::Fixed { step: 1.0 },
        ..Default::default()
    };
    for order in [10, 12, 13, 14, 20] {
        let e = order_test(&seqs, &opts, order, Method::G).unwrap_err();
        assert!(e.to_string().contains("histories"), "order {order}: {e}");
    }
    let t = order_test(&seqs, &opts, 5, Method::G).unwrap();
    assert_eq!(t.expected.unwrap().shape(), (1024, 4));
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn order_test_with_more_histories_than_transitions_raises_before_allocating() {
    assert_bounded("inner_order_test_order_beyond_the_data", 2_000_000);
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_exponential_test_many_replicates() {
    let so = Sojourn {
        n_beds: vec![2],
        n_open: vec![0],
        total: vec![3.0],
        mean_length: vec![1.5],
        thickness: vec![vec![1.0, 2.0]],
        open: vec![vec![false, false]],
    };
    let p = so.exponential_test(0, 60_000_000, 7).unwrap().p_value;
    assert!(p > 0.0 && p <= 1.0, "{p}");
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn exponential_test_replicates_take_constant_memory() {
    assert_bounded("inner_exponential_test_many_replicates", 300_000);
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_simulate_huge_n() {
    let p = TransitionMatrix::new(Mat::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5])).unwrap();
    let c =
        MarkovChain::from_transition(p, Kind::Fixed { step: 1.0 }, None, Direction::Up).unwrap();
    let r = std::panic::catch_unwind(|| {
        c.simulate(
            Target::Steps(4),
            &Hard::default(),
            Start::Stationary,
            1usize << 62,
            0,
        )
        .map(|_| ())
    });
    assert!(matches!(r, Ok(Err(_))), "n = 2^62, 4 steps: {r:?}");
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn simulate_with_huge_n_raises_instead_of_panicking() {
    assert_bounded("inner_simulate_huge_n", 2_000_000);
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_length_simulation_huge_length() {
    let c = MarkovChain::from_rates(golden_q(), Direction::Up).unwrap();
    let r = c.simulate(
        Target::Length(1e12),
        &Hard::default(),
        Start::Stationary,
        1,
        0,
    );
    assert!(
        r.is_err(),
        "about 3e11 beds requested and the call returned Ok"
    );
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn length_simulation_validates_size_before_allocating() {
    assert_bounded("inner_length_simulation_huge_length", 1_000_000);
}

#[test]
#[ignore = "run by its bounded parent"]
fn inner_long_discrete_bridge() {
    let len = 30_000_000usize;
    let p = TransitionMatrix::new(Mat::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8])).unwrap();
    let c =
        MarkovChain::from_transition(p, Kind::Fixed { step: 1.0 }, None, Direction::Up).unwrap();
    let last = (len - 1) as f64;
    let hard = Hard {
        from: vec![0.0, last],
        to: vec![0.0, last],
        codes: vec![0, 1],
    };
    let Simulated::Codes(x) = c
        .simulate(Target::Steps(len), &hard, Start::Stationary, 1, 0)
        .unwrap()
    else {
        panic!("steps give codes")
    };
    assert_eq!((x[0][0], x[0][len - 1]), (0, 1));
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "macOS has no address-space cap (ulimit -v) and no timeout(1)"
)]
fn discrete_bridge_reuses_converged_powers() {
    // chain.md 9: P^r stops at convergence and is reused, so a 3e7-step bridge needs the
    // 3e7-entry path buffers (about 360 MB) and not 3e7 stored columns.
    assert_bounded("inner_long_discrete_bridge", 1_200_000);
}

#[test]
fn length_mode_rejects_two_codes_at_one_point() {
    let c = MarkovChain::from_rates(golden_q(), Direction::Up).unwrap();
    let hard = Hard {
        from: vec![5.0, 5.0],
        to: vec![5.0, 5.0],
        codes: vec![0, 1],
    };
    let r = c.simulate(Target::Length(10.0), &hard, Start::Stationary, 1, 0);
    assert!(
        r.is_err(),
        "two hard codes at position 5 were accepted: {r:?}"
    );
    let p = golden_q().transition(1.0);
    let f =
        MarkovChain::from_transition(p, Kind::Fixed { step: 1.0 }, None, Direction::Up).unwrap();
    assert!(
        f.simulate(Target::Steps(10), &hard, Start::Stationary, 1, 0)
            .is_err()
    );
}

#[test]
fn sojourn_mean_length_matches_embedded_fit() {
    let holes: Vec<String> = ["a", "a", "a", "a", "a", "a", "b", "b", "b"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let from = [0.0, 1.0, 3.0, 4.0, 5.0, 7.0, 0.0, 2.0, 3.0];
    let to = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 2.0, 3.0, 4.0];
    let codes = [0, 1, 0, MISSING, 2, 1, 2, 0, 1];
    let seqs = SequenceSet::from_intervals(&holes, &from, &to, &codes, 3).unwrap();
    let opts = FitOptions {
        on_empty_row: EmptyRow::Uniform,
        ..Default::default()
    };
    let c = MarkovChain::fit(&seqs, &opts).unwrap();
    let s = sojourn(&seqs, opts.direction, opts.gap);
    let n = c.counts.as_ref().unwrap();
    let t = c.exposure.as_ref().unwrap();
    for j in 0..3 {
        assert!((s.total[j] - t[j]).abs() < 1e-12, "T[{j}]");
        let exits: f64 = n.row(j).sum();
        let want = if exits > 0.0 { t[j] / exits } else { f64::NAN };
        let got = s.mean_length[j];
        assert!(
            (got - want).abs() < 1e-12 || (got.is_nan() && want.is_nan()),
            "category {j}: sojourn mean_length {got}, T_j / exits_j = {want}"
        );
    }
}

#[test]
fn fixed_order1_and_independence_test_agree() {
    let mut s: Vec<Code> = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        .iter()
        .copied()
        .cycle()
        .take(200)
        .collect();
    s.push(2);
    let seqs = SequenceSet::from_regular(&[s], 3).unwrap();
    let opts = FitOptions {
        kind: Kind::Fixed { step: 1.0 },
        direction: Direction::Down,
        on_empty_row: EmptyRow::Uniform,
        ..Default::default()
    };
    let a = MarkovChain::fit(&seqs, &opts)
        .unwrap()
        .independence_test(Method::G)
        .unwrap();
    let b = order_test(&seqs, &opts, 1, Method::G).unwrap();
    assert_eq!(
        (a.df, a.statistic),
        (b.df, b.statistic),
        "independence_test and order_test(1) differ on the same counts"
    );
}

#[test]
fn from_geology_rejects_proportions_not_summing_to_one() {
    let mut pi = Mat::zeros(3, 3);
    pi[(0, 1)] = 0.6;
    pi[(1, 0)] = 0.7;
    let r = MarkovChain::from_geology(
        &[0.5, 0.5, 0.5],
        &[4.0, 2.0, 1.0],
        Some(&pi),
        2,
        false,
        Direction::Up,
    );
    assert!(
        r.is_err(),
        "proportions summing to 1.5: {:?}",
        r.map(|c| c.proportions)
    );
}

#[test]
fn fixed_fit_with_a_category_only_at_the_top_of_a_hole() {
    let mut s: Vec<Code> = vec![2];
    s.extend(
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
            .iter()
            .copied()
            .cycle()
            .take(200),
    );
    let seqs = SequenceSet::from_regular(&[s], 3).unwrap();
    let opts = FitOptions {
        kind: Kind::Fixed { step: 1.0 },
        direction: Direction::Down,
        ..Default::default()
    };
    let c = MarkovChain::fit(&seqs, &opts);
    assert!(
        c.is_ok(),
        "category 2 is transient (row 2 = (1, 0, 0), column 2 empty): the stationary law is unique with p_2 = 0, but fit failed: {:?}",
        c.err()
    );
    assert_eq!(c.unwrap().proportions[2], 0.0);
}

#[test]
fn sojourn_with_nan_gap_does_not_panic() {
    let seqs = no_repeat_sequence();
    let r = std::panic::catch_unwind(|| sojourn(&seqs, Direction::Up, f64::NAN));
    assert!(r.is_ok(), "sojourn with gap NaN panicked");
}
