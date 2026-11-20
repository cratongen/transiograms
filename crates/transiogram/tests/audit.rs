use transiograms_core::synthetic::rate_holes;
use transiograms_core::{Direction, Mat, RateMatrix, SequenceSet};
use transiograms_transiogram::{Model1D, Structure, downhole};

const P: [f64; 3] = [0.5, 0.3, 0.2];

fn carle() -> RateMatrix {
    let partial = Mat::from_row_slice(3, 3, &[-0.25, 0.15, 0.0, 0.35, -0.5, 0.0, 0.0, 0.0, 0.0]);
    RateMatrix::from_partial(&partial, &P, 2).unwrap()
}

fn assert_sill(model: &Model1D, p: &[f64]) {
    for h in [1e4, f64::INFINITY] {
        let t = model.transition(h);
        for i in 0..3 {
            for j in 0..3 {
                assert!((t[(i, j)] - p[j]).abs() < 1e-9, "h {h} ({i},{j})");
            }
        }
    }
}

/// Release audit: fitted models on short seeded logs honour the given proportions at the sill
/// and, for the Markov fit, the mean-length relation `L_i = -1 / r_ii` at the origin.
#[test]
fn fitted_models_honour_sill_and_origin_on_short_logs() {
    let lags: Vec<f64> = (1..=6).map(f64::from).collect();
    for seed in 0..10u64 {
        let iv = rate_holes(&carle(), &[60.0; 5], Direction::Down, seed).unwrap();
        let seqs = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
        let logs = SequenceSet::from_regular(&seqs.regular(Direction::Down, 1.0, 0.0).unwrap(), 3)
            .unwrap();
        let exp = downhole(&logs, Direction::Down, &lags, 0.0).unwrap();
        let total: f64 = exp.tail_totals[0].iter().sum();
        let emp: Vec<f64> = exp.tail_totals[0].iter().map(|n| n / total).collect();

        let fit = Model1D::fit_markov(&exp, Some(&emp), 2, None).unwrap();
        assert_sill(&fit, &emp);
        let Model1D::Markov { rates, .. } = &fit else {
            panic!()
        };
        let eps = 1e-6;
        let t = fit.transition(eps);
        for i in 0..3 {
            let slope = (t[(i, i)] - 1.0) / eps;
            assert!(
                (slope - rates.matrix()[(i, i)]).abs() < 1e-4,
                "seed {seed} row {i}"
            );
        }

        let ent =
            Model1D::fit_entrywise(&exp, &emp, &[Structure::Exponential], &[0, 1, 2]).unwrap();
        assert_sill(&ent, &emp);
        let t = ent.transition(lags[0]);
        for i in 0..3 {
            assert!((t.row(i).sum() - 1.0).abs() < 1e-12);
            assert!(t.row(i).iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
    }
}
