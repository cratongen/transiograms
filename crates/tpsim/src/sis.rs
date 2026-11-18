use rayon::prelude::*;
use varvite_core::rng::tag;
use varvite_core::{Code, Grid, MISSING, Realizations, Result, Snapped, Stream, invalid, memory};
use varvite_transiogram::TransitionModel;

use crate::Options;
use crate::cokriging::{cokrige_offsets, correct};
use crate::quench::quench;
use crate::search::Neighbourhood;

/// Peak bytes of one quench beside the codes: the cell list, its shuffled copy, the lags and the
/// pair counts (rebuilt once, so held twice), for at most 13 lags.
fn quench_bytes(k: usize, cells: u64) -> u64 {
    let k = k as u64;
    cells
        .saturating_mul(2 * 8)
        .saturating_add(13 * (4 * k * k * 8 + 2 * k * 8 + 64) + 4 * k * 8 + (1 << 10))
}

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
    let active = mask.map_or(n_cells, |m| m.iter().filter(|&&a| a).count());
    let path_len = active - hard.cells.len();
    let in_flight = n.min(rayon::current_num_threads()) as u64;
    let per_realization = (path_len as u64).saturating_mul(8).saturating_add(
        opts.quench
            .as_ref()
            .map_or(0, |_| quench_bytes(k, path_len as u64)),
    );
    let history = opts.quench.as_ref().map_or(0, |q| {
        (q.max_iter as u64).saturating_add(1).saturating_mul(3 * 8)
    });
    let per_output =
        (size_of::<Result<(f64, Vec<f64>)>>() + size_of::<f64>() + size_of::<Vec<f64>>()) as u64
            + history;
    let held = (n as u64)
        .saturating_mul(n_cells as u64)
        .saturating_mul(size_of::<Code>() as u64)
        .saturating_add(n_cells as u64)
        .saturating_add((n as u64).saturating_mul(per_output))
        .saturating_add(in_flight.saturating_mul(per_realization));
    let budget = opts.memory.or_else(memory::available_bytes);
    memory::check(held, budget, "tprogs_simulate realizations")?;
    let (table, max_data) = crate::search_table(model, grid, opts, budget, held)?;
    let nb = Neighbourhood::new(model, grid, &table);
    let mut frozen = vec![false; n_cells];
    for &cell in &hard.cells {
        frozen[cell] = true;
    }
    let Some(total) = n.checked_mul(n_cells) else {
        return invalid(format!(
            "{n} realizations of {n_cells} cells overflow the address space"
        ));
    };
    let mut codes = vec![MISSING; total];
    let runs: Vec<Result<(f64, Vec<f64>)>> = codes
        .par_chunks_mut(n_cells)
        .enumerate()
        .map(|(r, codes)| {
            let r = r as u64;
            for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
                codes[cell] = code;
            }
            let mut order = Vec::with_capacity(path_len);
            order.extend((0..n_cells).filter(|&c| !frozen[c] && mask.is_none_or(|m| m[c])));
            Stream::new(opts.seed, &[r, tag::PATH]).shuffle(&mut order);
            let mut draw = Stream::new(opts.seed, &[r, tag::DRAW]);
            let mut data = Vec::with_capacity(max_data);
            let mut corrected = 0usize;
            for &cell in &order {
                nb.search(cell, codes, max_data, opts.max_per_octant, &mut data);
                let mut pr = cokrige_offsets(&nb, &data, opts.svd_cutoff)?;
                if correct(&mut pr) {
                    corrected += 1;
                }
                codes[cell] = draw.categorical(&pr) as Code;
            }
            let fraction = if order.is_empty() {
                0.0
            } else {
                corrected as f64 / order.len() as f64
            };
            drop(order);
            let objective = match &opts.quench {
                Some(q) => quench(codes, &frozen, grid, mask, model, q, opts.seed, r)?,
                None => vec![],
            };
            Ok((fraction, objective))
        })
        .collect();
    let mut corrected = Vec::with_capacity(n);
    let mut objective = Vec::with_capacity(n);
    for run in runs {
        let (c, o) = run?;
        corrected.push(c);
        objective.push(o);
    }
    Ok(Realizations {
        n,
        n_cells,
        k,
        codes,
        corrected,
        objective,
    })
}
