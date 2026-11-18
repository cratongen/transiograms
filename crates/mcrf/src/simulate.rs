use rayon::prelude::*;
use varvite_core::rng::tag;
use varvite_core::{Code, Grid, MISSING, Mat, Realizations, Result, Snapped, Stream, memory};
use varvite_transiogram::TransitionModel;

use crate::Options;
use crate::conditional::resolve;
use crate::search::Search;

pub fn simulate(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Realizations> {
    let frozen = crate::hard_codes(model, grid, mask, hard)?;
    let k = model.k();
    let n_cells = grid.n_cells();
    let n = opts.n_realizations;
    let concurrent = rayon::current_num_threads().min(n) as u64;
    memory::check(
        n as u64 * n_cells as u64 * 4 + concurrent * n_cells as u64 * 8,
        opts.memory,
        "MCRF simulation",
    )?;
    let table = model.table(grid, opts.closeness_limit, opts.memory)?;
    let search = Search::new(&table, grid, opts.neighbourhood);
    let p = &model.proportions;
    let path: Vec<usize> = (0..n_cells)
        .filter(|&c| frozen[c] == MISSING && mask.is_none_or(|m| m[c]))
        .collect();

    let mut out = vec![MISSING; n * n_cells];
    out.par_chunks_mut(n_cells.max(1))
        .enumerate()
        .for_each(|(r, codes)| {
            let r = r as u64;
            codes.copy_from_slice(&frozen);
            let mut order = path.clone();
            Stream::new(opts.seed, &[r, tag::PATH]).shuffle(&mut order);
            let mut draw = Stream::new(opts.seed, &[r, tag::DRAW]);
            let mut found: Vec<(usize, Code)> = Vec::new();
            let mut terms: Vec<(&Mat, Code)> = Vec::new();
            let mut probabilities = vec![0.0; k];
            for cell in order {
                search.find(grid.ijk(cell), codes, &mut found);
                terms.clear();
                terms.extend(found.iter().map(|&(t, c)| (&table.transition[t], c)));
                resolve(p, &terms, &mut probabilities);
                codes[cell] = draw.categorical(&probabilities) as Code;
            }
        });

    Ok(Realizations {
        n,
        n_cells,
        k,
        codes: out,
        corrected: vec![0.0; n],
        objective: Vec::new(),
    })
}
