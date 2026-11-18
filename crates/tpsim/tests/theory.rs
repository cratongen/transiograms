use varvite_core::rng::tag;
use varvite_core::synthetic::markov_codes;
use varvite_core::*;
use varvite_tpsim::*;
use varvite_transiogram::TransitionModel;

fn carle_embedded() -> Mat {
    Mat::from_row_slice(
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
    )
}

/// The M5 model: K = 3, lateral maximum entropy, Carle embedded chain vertically.
fn m5_model() -> TransitionModel {
    let p = [0.5, 0.3, 0.2];
    let lateral = [12.0, 8.0, 6.0];
    let vertical = [2.0, 1.0, 0.8];
    let pi = carle_embedded();
    TransitionModel::from_geology(
        &p,
        [&lateral, &lateral, &vertical],
        [None, None, Some(&pi)],
        2,
        [0.0; 3],
    )
    .unwrap()
}

/// An asymmetric, rotated model for the cokriging identities.
fn rotated_model() -> TransitionModel {
    let p = [0.5, 0.3, 0.2];
    let pi = carle_embedded();
    TransitionModel::from_geology(
        &p,
        [&[9.0, 6.0, 5.0], &[5.0, 4.0, 3.0], &[4.0, 2.0, 1.0]],
        [None, Some(&pi), Some(&pi)],
        2,
        [30.0, 10.0, 5.0],
    )
    .unwrap()
}

/// Vertical holes through `columns`, each a stationary draw of the model's z chain at the cell
/// size, bottom to top, snapped onto the grid.
fn vertical_holes(model: &TransitionModel, grid: &Grid, columns: &[[usize; 2]]) -> Snapped {
    let t = TransitionMatrix::new(model.transition([0.0, 0.0, grid.size[2]])).unwrap();
    let mut coords = vec![];
    let mut codes = vec![];
    for (h, &[i, j]) in columns.iter().enumerate() {
        let column = markov_codes(
            &t,
            None,
            grid.count[2],
            &mut Stream::new(0, &[tag::SAMPLE, h as u64]),
        )
        .unwrap();
        for (kz, &c) in column.iter().enumerate() {
            coords.push(grid.centroid(grid.index([i, j, kz])));
            codes.push(c);
        }
    }
    grid.snap(&coords, &codes, None).unwrap()
}

fn small_case() -> (TransitionModel, Grid, Snapped) {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], [12, 10, 8]).unwrap();
    let hard = vertical_holes(&model, &grid, &[[2, 2], [9, 3], [5, 8]]);
    (model, grid, hard)
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn single_datum_gives_the_transition_row() {
    let model = rotated_model();
    let x1 = [1.0, -2.0, 0.5];
    let targets = [
        [3.0, 1.0, 1.5],
        [-4.0, 0.5, -0.7],
        [1.0, -2.0, 2.5],
        [0.2, 9.0, 0.0],
    ];
    for x0 in targets {
        let t = model.transition([x0[0] - x1[0], x0[1] - x1[1], x0[2] - x1[2]]);
        for c in 0..3u32 {
            let pr = cokrige(&model, &[(x1, c)], x0, 1e-8).unwrap();
            let row: Vec<f64> = (0..3).map(|j| t[(c as usize, j)]).collect();
            assert!(max_abs(&pr, &row) < 1e-12, "{pr:?} vs {row:?}");
        }
    }
}

#[test]
fn probabilities_sum_to_one_before_correction() {
    let model = rotated_model();
    let mut s = Stream::new(11, &[0]);
    let mut corrected = 0;
    for trial in 0..200 {
        let n = 1 + trial % 12;
        let scale = if trial % 2 == 0 { 1.0 } else { 0.2 };
        let data: Vec<([f64; 3], Code)> = (0..n)
            .map(|_| {
                let x = [
                    scale * (20.0 * s.uniform() - 10.0),
                    scale * (20.0 * s.uniform() - 10.0),
                    scale * (4.0 * s.uniform() - 2.0),
                ];
                (x, s.below(3) as Code)
            })
            .collect();
        let mut pr = cokrige(&model, &data, [0.0; 3], 1e-8).unwrap();
        let total: f64 = pr.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "trial {trial}: sum {total}");
        if correct(&mut pr) {
            corrected += 1;
        }
        assert!(pr.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    assert!(
        corrected > 0,
        "the random configurations never needed correction"
    );
}

#[test]
fn no_data_gives_proportions_and_bad_codes_raise() {
    let model = rotated_model();
    assert_eq!(
        cokrige(&model, &[], [0.0; 3], 1e-8).unwrap(),
        model.proportions
    );
    assert!(cokrige(&model, &[([0.0; 3], 3)], [1.0; 3], 1e-8).is_err());
    assert!(cokrige(&model, &[([0.0; 3], 0)], [1.0; 3], -1.0).is_err());
}

#[test]
fn prediction_is_a_distribution_and_honours_hard_data() {
    let (model, grid, hard) = small_case();
    let mut mask = vec![true; grid.n_cells()];
    mask[grid.index([11, 9, 7])] = false;
    let opts = Options::default();
    let pred = predict(&model, &grid, Some(&mask), &hard, &opts).unwrap();
    assert_eq!(pred.probabilities.len(), grid.n_cells() * 3);
    for cell in 0..grid.n_cells() {
        let row = &pred.probabilities[cell * 3..cell * 3 + 3];
        if !mask[cell] {
            assert!(row.iter().all(|p| p.is_nan()));
            assert_eq!(pred.most_likely[cell], MISSING);
            continue;
        }
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)), "{row:?}");
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        assert_eq!(pred.probabilities[cell * 3 + code as usize], 1.0);
        assert_eq!(pred.most_likely[cell], code);
    }
    assert!((0.0..=1.0).contains(&pred.corrected));
}

#[test]
fn invalid_options_raise() {
    let (model, grid, hard) = small_case();
    let odd = Options {
        max_data: 5,
        ..Options::default()
    };
    assert!(simulate(&model, &grid, None, &hard, &odd).is_err());
    let mask = vec![false; grid.n_cells()];
    assert!(predict(&model, &grid, Some(&mask), &hard, &Options::default()).is_err());
    assert!(predict(&model, &grid, Some(&[true]), &hard, &Options::default()).is_err());
    let tiny = Options {
        memory: Some(16),
        ..Options::default()
    };
    let e = simulate(&model, &grid, None, &hard, &tiny)
        .unwrap_err()
        .to_string();
    assert!(e.contains("bytes"), "{e}");
}

#[test]
fn simulation_honours_hard_data_and_quench_never_increases() {
    let (model, grid, hard) = small_case();
    let mut mask = vec![true; grid.n_cells()];
    for i in 0..4 {
        mask[grid.index([i, 0, 0])] = false;
    }
    let opts = Options {
        n_realizations: 3,
        seed: 7,
        quench: Some(QuenchOptions {
            max_iter: 6,
            tol: 0.0,
            ..QuenchOptions::default()
        }),
        ..Options::default()
    };
    let sims = simulate(&model, &grid, Some(&mask), &hard, &opts).unwrap();
    assert_eq!((sims.n, sims.n_cells, sims.k), (3, grid.n_cells(), 3));
    for r in 0..3 {
        let codes = &sims.codes[r * grid.n_cells()..(r + 1) * grid.n_cells()];
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            assert_eq!(codes[cell], code, "realization {r} cell {cell}");
        }
        for cell in 0..grid.n_cells() {
            assert_eq!(codes[cell] == MISSING, !mask[cell]);
        }
        let o = &sims.objective[r];
        assert!(o.len() >= 2);
        assert!(o.windows(2).all(|w| w[1] <= w[0]), "{o:?}");
        assert!(o.last().unwrap() < &o[0]);
        let lags = quench_lags(&model, &grid, 0.4);
        let recomputed = objective(codes, &grid, Some(&mask), &model, &lags, 1.0);
        assert!((recomputed - o.last().unwrap()).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&sims.corrected[r]));
    }
    let again = simulate(&model, &grid, Some(&mask), &hard, &opts).unwrap();
    assert_eq!(again, sims);
    let other = simulate(
        &model,
        &grid,
        Some(&mask),
        &hard,
        &Options { seed: 8, ..opts },
    )
    .unwrap();
    assert_ne!(other.codes, sims.codes);
}

#[test]
fn quench_lags_in_2d_and_3d() {
    let model = m5_model();
    let g3 = Grid::new([0.0; 3], [1.0, 1.0, 0.25], [5, 5, 5]).unwrap();
    let all = quench_lags(&model, &g3, 1e-9);
    assert_eq!(all.len(), 13);
    for d in &all {
        assert!(!all.contains(&[-d[0], -d[1], -d[2]]));
    }
    let strict = quench_lags(&model, &g3, 0.999);
    assert_eq!(strict, vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    let g2 = Grid::new([0.0; 3], [1.0, 1.0, 1.0], [5, 5, 1]).unwrap();
    assert_eq!(quench_lags(&model, &g2, 1e-9).len(), 4);
}

#[test]
fn threads_do_not_change_results() {
    let (model, grid, hard) = small_case();
    let opts = Options {
        n_realizations: 4,
        seed: 3,
        ..Options::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (
                    simulate(&model, &grid, None, &hard, &opts).unwrap(),
                    predict(&model, &grid, None, &hard, &opts).unwrap(),
                )
            })
    };
    let (s1, p1) = run(1);
    let (s4, p4) = run(4);
    assert_eq!(s1.codes, s4.codes);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&s1.corrected), bits(&s4.corrected));
    for (a, b) in s1.objective.iter().zip(&s4.objective) {
        assert_eq!(bits(a), bits(b));
    }
    assert_eq!(bits(&p1.probabilities), bits(&p4.probabilities));
}

/// tpsim.md section 5: `Var(p_hat_k) ~ |D|^-2 sum_x sum_y C_kk(y - x)` over the table lags.
fn proportion_sd(model: &TransitionModel, grid: &Grid, limit: f64) -> Vec<f64> {
    let table = model.table(grid, limit, None).unwrap();
    let p = &model.proportions;
    let n = grid.n_cells() as f64;
    (0..model.k())
        .map(|k| {
            let mut sum = n * p[k] * (1.0 - p[k]);
            for (d, t) in table.offsets.iter().zip(&table.transition) {
                let pairs: f64 = (0..3)
                    .map(|a| (grid.count[a] as i64 - d[a].abs()) as f64)
                    .product();
                sum += pairs * p[k] * (t[(k, k)] - p[k]);
            }
            (sum / (n * n)).sqrt()
        })
        .collect()
}

/// Ensemble-mean auto-transiograms along one axis for lags 1..=n_lags.
fn auto_transiograms(
    sims: &Realizations,
    grid: &Grid,
    axis: usize,
    n_lags: usize,
) -> Vec<Vec<f64>> {
    let k = sims.k;
    let mut mean = vec![vec![0.0; k]; n_lags];
    let mut used = vec![vec![0usize; k]; n_lags];
    for r in 0..sims.n {
        let codes = &sims.codes[r * sims.n_cells..(r + 1) * sims.n_cells];
        for l in 1..=n_lags {
            let mut d = [0i64; 3];
            d[axis] = l as i64;
            let mut f = vec![0.0; k];
            let mut rows = vec![0.0; k];
            for x in 0..sims.n_cells {
                if let Some(y) = grid.offset(x, d) {
                    let a = codes[x] as usize;
                    rows[a] += 1.0;
                    if codes[y] == codes[x] {
                        f[a] += 1.0;
                    }
                }
            }
            for c in 0..k {
                if rows[c] > 0.0 {
                    mean[l - 1][c] += f[c] / rows[c];
                    used[l - 1][c] += 1;
                }
            }
        }
    }
    for l in 0..n_lags {
        for c in 0..k {
            mean[l][c] /= used[l][c] as f64;
        }
    }
    mean
}

/// M5 acceptance (tpsim.md section 6, sizes as recorded under "Implementer decisions").
#[test]
fn m5_acceptance() {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], M5_SIZE, M5_COUNT).unwrap();
    let sd = proportion_sd(&model, &grid, 0.01);
    for (k, s) in sd.iter().enumerate() {
        assert!(3.0 * s < 0.02, "category {k}: 3 sigma = {}", 3.0 * s);
    }
    let [nx, ny, _] = grid.count;
    let columns: Vec<[usize; 2]> = (0..16)
        .map(|h| [(2 * (h % 4) + 1) * nx / 8, (2 * (h / 4) + 1) * ny / 8])
        .collect();
    let hard = vertical_holes(&model, &grid, &columns);
    let opts = Options {
        n_realizations: 20,
        seed: 0,
        ..Options::default()
    };
    let sims = simulate(&model, &grid, None, &hard, &opts).unwrap();
    let n = grid.n_cells();
    for r in 0..sims.n {
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            assert_eq!(
                sims.codes[r * n + cell],
                code,
                "realization {r} cell {cell}"
            );
        }
    }
    let mut counts = [0.0; 3];
    for &c in &sims.codes {
        counts[c as usize] += 1.0;
    }
    for k in 0..3 {
        let p = counts[k] / sims.codes.len() as f64;
        eprintln!(
            "category {k}: ensemble proportion {p:.4}, target {}",
            model.proportions[k]
        );
        assert!(
            (p - model.proportions[k]).abs() < 0.02,
            "category {k}: ensemble proportion {p}"
        );
    }
    let lengths = [
        [12.0, 8.0, 6.0],
        [12.0, 8.0, 6.0],
        model.mean_lengths([0.0, 0.0, 1.0]).try_into().unwrap(),
    ];
    for axis in 0..3 {
        let longest = lengths[axis].iter().copied().fold(0.0, f64::max);
        let n_lags = ((2.0 * longest / grid.size[axis]) + 1e-9).floor() as usize;
        let n_lags = n_lags.min(grid.count[axis] - 1);
        let exp = auto_transiograms(&sims, &grid, axis, n_lags);
        let mut sq = 0.0;
        for l in 1..=n_lags {
            let mut d = [0i64; 3];
            d[axis] = l as i64;
            let t = model.transition(grid.lag(d));
            for k in 0..3 {
                sq += (exp[l - 1][k] - t[(k, k)]).powi(2);
            }
        }
        let rmse = (sq / (3 * n_lags) as f64).sqrt();
        eprintln!("axis {axis}: auto-transiogram RMSE {rmse:.4} over {n_lags} lags");
        assert!(
            rmse < 0.05,
            "axis {axis}: auto-transiogram RMSE {rmse} over {n_lags} lags"
        );
    }
}

const M5_SIZE: [f64; 3] = [6.0, 6.0, 1.0];
const M5_COUNT: [usize; 3] = [64, 64, 32];
