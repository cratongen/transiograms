//! Seeded synthetic data with known parameters, for every crate's tests: discrete and
//! continuous Markov chains, hidden Markov samples and layered 3D fields with drillholes.
//! Sequence or hole `i` draws from `Stream::new(seed, &[tag::SAMPLE, i])`, so results do not
//! depend on the thread count.

use rayon::prelude::*;

use crate::codes::Code;
use crate::error::{Result, invalid};
use crate::grid::Grid;
use crate::matrix::{RateMatrix, TransitionMatrix};
use crate::rng::{Stream, tag};
use crate::sequence::Direction;

fn first_state(start: Option<Code>, law: &[f64], stream: &mut Stream) -> Result<usize> {
    match start {
        Some(c) if (c as usize) < law.len() => Ok(c as usize),
        Some(c) => invalid(format!(
            "start code {c} is not one of {} categories",
            law.len()
        )),
        None => Ok(stream.categorical(law)),
    }
}

/// `n` steps of the chain `p`, from `start` or from its stationary law.
pub fn markov_codes(
    p: &TransitionMatrix,
    start: Option<Code>,
    n: usize,
    stream: &mut Stream,
) -> Result<Vec<Code>> {
    let law = match start {
        Some(_) => vec![1.0; p.k()],
        None => p.stationary()?,
    };
    let m = p.matrix();
    let mut state = first_state(start, &law, stream)?;
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        if t > 0 {
            let row: Vec<f64> = m.row(state).iter().copied().collect();
            state = stream.categorical(&row);
        }
        out.push(state as Code);
    }
    Ok(out)
}

/// One stationary sequence per entry of `lengths`.
pub fn markov_holes(p: &TransitionMatrix, lengths: &[usize], seed: u64) -> Result<Vec<Vec<Code>>> {
    p.stationary()?;
    lengths
        .par_iter()
        .enumerate()
        .map(|(i, &n)| markov_codes(p, None, n, &mut Stream::new(seed, &[tag::SAMPLE, i as u64])))
        .collect()
}

/// Beds of a continuous-lag chain on positions `[0, length)`: `(start, end, code)` per bed,
/// the first drawn from `start` or the stationary law of `q`, the last cut at `length`.
pub fn rate_beds(
    q: &RateMatrix,
    length: f64,
    start: Option<Code>,
    stream: &mut Stream,
) -> Result<Vec<(f64, f64, Code)>> {
    if !(length > 0.0 && length.is_finite()) {
        return invalid(format!("length must be positive and finite, got {length}"));
    }
    let law = match start {
        Some(_) => vec![1.0; q.k()],
        None => q.stationary()?,
    };
    let embedded = q.embedded();
    let lengths = q.mean_lengths();
    let mut state = first_state(start, &law, stream)?;
    let mut position = 0.0;
    let mut out = Vec::new();
    loop {
        let thickness = if q.k() == 1 {
            f64::INFINITY
        } else {
            lengths[state] * stream.exponential()
        };
        let end = (position + thickness).min(length);
        out.push((position, end, state as Code));
        if end >= length {
            return Ok(out);
        }
        position = end;
        let row: Vec<f64> = embedded.matrix().row(state).iter().copied().collect();
        state = stream.categorical(&row);
    }
}

/// Top-down intervals, as a user would pass them.
#[derive(Clone, Debug, PartialEq)]
pub struct Intervals {
    pub holes: Vec<String>,
    pub from: Vec<f64>,
    pub to: Vec<f64>,
    pub codes: Vec<Code>,
}

/// One hole per entry of `lengths`, simulated along `direction` from the stationary law of `q`
/// and returned top-down by depth (hole ids "0", "1", ...). Fitting with the same direction
/// recovers `q`.
pub fn rate_holes(
    q: &RateMatrix,
    lengths: &[f64],
    direction: Direction,
    seed: u64,
) -> Result<Intervals> {
    let holes: Vec<Vec<(f64, f64, Code)>> = lengths
        .par_iter()
        .enumerate()
        .map(|(i, &l)| rate_beds(q, l, None, &mut Stream::new(seed, &[tag::SAMPLE, i as u64])))
        .collect::<Result<_>>()?;
    let mut out = Intervals {
        holes: vec![],
        from: vec![],
        to: vec![],
        codes: vec![],
    };
    for (i, beds) in holes.into_iter().enumerate() {
        let length = lengths[i];
        let mut rows: Vec<(f64, f64, Code)> = match direction {
            Direction::Down => beds,
            Direction::Up => beds
                .into_iter()
                .map(|(a, b, c)| (length - b, length - a, c))
                .collect(),
        };
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (from, to, code) in rows {
            out.holes.push(i.to_string());
            out.from.push(from);
            out.to.push(to);
            out.codes.push(code);
        }
    }
    Ok(out)
}

/// Emission law of one hidden state.
#[derive(Clone, Debug, PartialEq)]
pub enum Emitter {
    /// Independent normal channels.
    Gaussian { mean: Vec<f64>, sd: Vec<f64> },
    /// One channel of codes `0..len`, stored as f64.
    Categorical { probabilities: Vec<f64> },
}

impl Emitter {
    fn dim(&self) -> usize {
        match self {
            Emitter::Gaussian { mean, .. } => mean.len(),
            Emitter::Categorical { .. } => 1,
        }
    }

    fn draw(&self, stream: &mut Stream, out: &mut Vec<f64>) {
        match self {
            Emitter::Gaussian { mean, sd } => {
                out.extend(mean.iter().zip(sd).map(|(m, s)| m + s * stream.normal()))
            }
            Emitter::Categorical { probabilities } => {
                out.push(stream.categorical(probabilities) as f64)
            }
        }
    }
}

/// Hidden states and observations (`t x d` row-major) per sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct HmmSample {
    pub d: usize,
    pub states: Vec<Vec<Code>>,
    pub observations: Vec<Vec<f64>>,
}

pub fn hmm_sample(
    start: &[f64],
    transition: &TransitionMatrix,
    emitters: &[Emitter],
    lengths: &[usize],
    seed: u64,
) -> Result<HmmSample> {
    let k = transition.k();
    if start.len() != k || emitters.len() != k {
        return invalid(format!(
            "give {k} start probabilities and {k} emitters, not {} and {}",
            start.len(),
            emitters.len()
        ));
    }
    let d = emitters[0].dim();
    if emitters.iter().any(|e| e.dim() != d) {
        return invalid("every emitter must have the same number of channels");
    }
    for e in emitters {
        match e {
            Emitter::Gaussian { mean, sd }
                if sd.len() != mean.len() || sd.iter().any(|s| s.is_nan() || *s < 0.0) =>
            {
                return invalid("Gaussian emitters need one non-negative sd per mean");
            }
            Emitter::Categorical { probabilities }
                if probabilities.iter().any(|p| p.is_nan() || *p < 0.0) =>
            {
                return invalid("categorical emitters need non-negative probabilities");
            }
            _ => {}
        }
    }
    let m = transition.matrix();
    let pieces: Vec<(Vec<Code>, Vec<f64>)> = lengths
        .par_iter()
        .enumerate()
        .map(|(i, &n)| {
            let mut stream = Stream::new(seed, &[tag::SAMPLE, i as u64]);
            let mut states = Vec::with_capacity(n);
            let mut obs = Vec::with_capacity(n * d);
            let mut s = 0;
            for t in 0..n {
                s = if t == 0 {
                    stream.categorical(start)
                } else {
                    let row: Vec<f64> = m.row(s).iter().copied().collect();
                    stream.categorical(&row)
                };
                states.push(s as Code);
                emitters[s].draw(&mut stream, &mut obs);
            }
            (states, obs)
        })
        .collect();
    let (states, observations) = pieces.into_iter().unzip();
    Ok(HmmSample {
        d,
        states,
        observations,
    })
}

/// A layer-cake field: one continuous-lag chain with generator `rates_z` along +z (up),
/// constant laterally, point-sampled at cell centres. Its exact transiogram is
/// `T(h) = exp(rates_z h_z)` for `h_z >= 0`, whatever `h_x` and `h_y`.
pub fn layered_field(grid: &Grid, rates_z: &RateMatrix, seed: u64) -> Result<Vec<Code>> {
    let [nx, ny, nz] = grid.count;
    let height = nz as f64 * grid.size[2];
    let beds = rate_beds(
        rates_z,
        height,
        None,
        &mut Stream::new(seed, &[tag::SAMPLE, 0]),
    )?;
    let mut column = Vec::with_capacity(nz);
    let mut b = 0;
    for kz in 0..nz {
        let z = (kz as f64 + 0.5) * grid.size[2];
        while b + 1 < beds.len() && beds[b].1 <= z {
            b += 1;
        }
        column.push(beds[b].2);
    }
    Ok((0..nx * ny * nz).map(|i| column[i / (nx * ny)]).collect())
}

/// Vertical drillholes through the given `(i, j)` columns of a field, one composite per cell,
/// top-down: point coordinates at centroids and intervals in depth (`depth = -z`).
#[derive(Clone, Debug, PartialEq)]
pub struct Drillholes {
    pub holes: Vec<String>,
    pub xyz: Vec<[f64; 3]>,
    pub from: Vec<f64>,
    pub to: Vec<f64>,
    pub codes: Vec<Code>,
}

pub fn drillholes(grid: &Grid, field: &[Code], columns: &[[usize; 2]]) -> Result<Drillholes> {
    if field.len() != grid.n_cells() {
        return invalid(format!(
            "field has {} values for {} cells",
            field.len(),
            grid.n_cells()
        ));
    }
    let mut out = Drillholes {
        holes: vec![],
        xyz: vec![],
        from: vec![],
        to: vec![],
        codes: vec![],
    };
    for (h, &[i, j]) in columns.iter().enumerate() {
        if i >= grid.count[0] || j >= grid.count[1] {
            return invalid(format!("column ({i}, {j}) is outside the grid"));
        }
        for kz in (0..grid.count[2]).rev() {
            let cell = grid.index([i, j, kz]);
            let bottom = grid.origin[2] + kz as f64 * grid.size[2];
            out.holes.push(h.to_string());
            out.xyz.push(grid.centroid(cell));
            out.from.push(-(bottom + grid.size[2]));
            out.to.push(-bottom);
            out.codes.push(field[cell]);
        }
    }
    Ok(out)
}
