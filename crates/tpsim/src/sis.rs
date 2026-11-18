use rayon::prelude::*;
use varvite_core::rng::tag;
use varvite_core::{Code, Grid, MISSING, Realizations, Result, Snapped, Stream, invalid, memory};
use varvite_transiogram::TransitionModel;

use crate::Options;
use crate::cokriging::{cokrige_offsets, correct};
use crate::quench::quench;
use crate::search::Neighbourhood;

pub fn simulate(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Realizations> {
    let k = crate::check_inputs(model, grid, mask, hard, opts)?;
    let n = opts.n_realizations;
    if n == 0 {
        return invalid("n_realizations must be at least 1");
    }
    let n_cells = grid.n_cells();
    memory::check(
        (n as u64).saturating_mul(n_cells as u64).saturating_mul(8),
        opts.memory,
        "tprogs_simulate realizations",
    )?;
    let table = model.table(grid, opts.closeness_limit, opts.memory)?;
    let nb = Neighbourhood::new(model, grid, &table);
    let mut base = vec![MISSING; n_cells];
    let mut frozen = vec![false; n_cells];
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        base[cell] = code;
        frozen[cell] = true;
    }
    let path: Vec<usize> = (0..n_cells)
        .filter(|&c| !frozen[c] && mask.is_none_or(|m| m[c]))
        .collect();
    let runs: Vec<(Vec<Code>, f64, Vec<f64>)> = (0..n as u64)
        .into_par_iter()
        .map(|r| {
            let mut codes = base.clone();
            let mut order = path.clone();
            Stream::new(opts.seed, &[r, tag::PATH]).shuffle(&mut order);
            let mut draw = Stream::new(opts.seed, &[r, tag::DRAW]);
            let mut data = Vec::with_capacity(opts.max_data);
            let mut corrected = 0usize;
            for &cell in &order {
                nb.search(cell, &codes, opts.max_data, opts.max_per_octant, &mut data);
                let mut pr = cokrige_offsets(&nb, &data, opts.svd_cutoff)?;
                if correct(&mut pr) {
                    corrected += 1;
                }
                codes[cell] = draw.categorical(&pr) as Code;
            }
            let objective = match &opts.quench {
                Some(q) => quench(&mut codes, &frozen, grid, mask, model, q, opts.seed, r)?,
                None => vec![],
            };
            let fraction = if order.is_empty() {
                0.0
            } else {
                corrected as f64 / order.len() as f64
            };
            Ok((codes, fraction, objective))
        })
        .collect::<Result<_>>()?;
    let mut out = Realizations {
        n,
        n_cells,
        k,
        codes: Vec::with_capacity(n * n_cells),
        corrected: Vec::with_capacity(n),
        objective: Vec::with_capacity(n),
    };
    for (codes, corrected, objective) in runs {
        out.codes.extend(codes);
        out.corrected.push(corrected);
        out.objective.push(objective);
    }
    Ok(out)
}
