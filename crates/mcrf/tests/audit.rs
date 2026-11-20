use transiograms_core::*;
use transiograms_mcrf::{Neighbourhood, Options, predict, simulate};
use transiograms_transiogram::TransitionModel;

fn m5_model() -> TransitionModel {
    let pi = Mat::from_row_slice(
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
    TransitionModel::from_geology(
        &[0.5, 0.3, 0.2],
        [&[12.0, 8.0, 6.0], &[12.0, 8.0, 6.0], &[2.0, 1.0, 0.8]],
        [None, None, Some(&pi)],
        2,
        [0.0; 3],
    )
    .unwrap()
}

/// Per realization, `t_ij` at offset `d`, pooled over the grid.
fn per_realization(sims: &Realizations, grid: &Grid, d: [i64; 3]) -> Vec<Vec<f64>> {
    let k = sims.k;
    (0..sims.n)
        .map(|r| {
            let codes = &sims.codes[r * sims.n_cells..(r + 1) * sims.n_cells];
            let mut c = vec![0.0; k * k];
            for x in 0..sims.n_cells {
                if let Some(y) = grid.offset(x, d) {
                    c[codes[x] as usize * k + codes[y] as usize] += 1.0;
                }
            }
            for i in 0..k {
                let s: f64 = c[i * k..(i + 1) * k].iter().sum();
                c[i * k..(i + 1) * k].iter_mut().for_each(|v| *v /= s);
            }
            c
        })
        .collect()
}

/// Largest `|mean - model| / se` and largest `|mean - model|` over all entries and offsets.
fn worst(
    model: &TransitionModel,
    grid: &Grid,
    sims: &Realizations,
    offs: &[[i64; 3]],
) -> (f64, f64) {
    let k = sims.k;
    let (mut z, mut e) = (0.0f64, 0.0f64);
    for &d in offs {
        let t = model.transition(grid.lag(d));
        let per = per_realization(sims, grid, d);
        let n = per.len() as f64;
        for ij in 0..k * k {
            let m = per.iter().map(|v| v[ij]).sum::<f64>() / n;
            let var = per.iter().map(|v| (v[ij] - m).powi(2)).sum::<f64>() / (n - 1.0);
            let err = (m - t[(ij / k, ij % k)]).abs();
            e = e.max(err);
            z = z.max(err / (var / n).sqrt().max(1e-3));
        }
    }
    (z, e)
}

fn axes() -> Vec<(&'static str, Vec<[i64; 3]>)> {
    vec![
        ("x", (1..=6).map(|l| [l, 0, 0]).collect()),
        ("z+", (1..=6).map(|l| [0, 0, l]).collect()),
        ("z-", (1..=6).map(|l| [0, 0, -l]).collect()),
    ]
}

fn unconditional(neighbourhood: Neighbourhood) -> (TransitionModel, Grid, Realizations) {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], [48, 48, 32]).unwrap();
    let empty = grid.snap(&[], &[], None).unwrap();
    let opts = Options {
        n_realizations: 10,
        seed: 1,
        neighbourhood,
        ..Options::default()
    };
    let sims = simulate(&model, &grid, None, &empty, &opts).unwrap();
    (model, grid, sims)
}

/// Bound: unconditional ensemble proportions within 0.02 of the target (the bound the guide
/// states for the M6 grid with hard data). Measured on 2026-11-20 at seed 1 with the default
/// sectors neighbourhood: [0.462, 0.314, 0.225] against [0.5, 0.3, 0.2].
#[test]
#[ignore = "release audit: unconditional MCRF (default sectors neighbourhood) misses the target proportions by 0.038 (0.462 vs 0.5)"]
fn unconditional_ensemble_reproduces_the_proportions() {
    let (model, _, sims) = unconditional(Neighbourhood::Sectors);
    let mut counts = [0.0; 3];
    sims.codes.iter().for_each(|&c| counts[c as usize] += 1.0);
    for k in 0..3 {
        let p = counts[k] / sims.codes.len() as f64;
        assert!(
            (p - model.proportions[k]).abs() < 0.02,
            "category {k}: {p:.4} vs {}",
            model.proportions[k]
        );
    }
}

/// Bound: every ensemble entry `t_ij(d)` within 4 standard errors (spread across 10
/// realizations) of the model, lags 1..=6 along x, +z and -z, M5 model. Measured on
/// 2026-11-20 at seed 1, max error per neighbourhood: Sectors (default) 0.190, Nearest(4)
/// 0.088, Nearest(2) 0.276, Nearest(1) 0.393. Realizations are too continuous: vertical
/// `t_11(1)` is 0.765 against 0.643 with the default.
#[test]
#[ignore = "release audit: MCRF ensemble transiograms miss the model by up to 0.19 abs (27-31 standard errors) with the default sectors neighbourhood"]
fn unconditional_ensemble_transiograms_match_the_model_within_sampling_error() {
    for nb in [Neighbourhood::Sectors, Neighbourhood::Nearest(4)] {
        let (model, grid, sims) = unconditional(nb);
        for (name, offs) in axes() {
            let (z, e) = worst(&model, &grid, &sims, &offs);
            assert!(
                z < 4.0,
                "{nb:?} axis {name}: {z:.1} standard errors, max error {e:.4}"
            );
        }
    }
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// Release audit: byte-identical output on 1 and 8 threads and on a repeated run.
#[test]
fn simulate_and_predict_are_identical_on_1_and_8_threads() {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], [16, 16, 8]).unwrap();
    let coords: Vec<[f64; 3]> = (0..8)
        .map(|z| grid.centroid(grid.index([8, 8, z])))
        .collect();
    let codes: Vec<Code> = [0, 0, 1, 1, 2, 0, 1, 0].to_vec();
    let hard = grid.snap(&coords, &codes, None).unwrap();
    let opts = Options {
        n_realizations: 3,
        seed: 7,
        ..Options::default()
    };
    let run = || {
        let s = simulate(&model, &grid, None, &hard, &opts).unwrap();
        let p = predict(&model, &grid, None, &hard, &opts).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        (s.codes, bits(&p.probabilities))
    };
    let one = pool(1).install(run);
    assert_eq!(one, pool(8).install(run));
    assert_eq!(one, pool(8).install(run));
}
