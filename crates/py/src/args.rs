//! Argument coercion shared by every area module.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use transiograms_core::{Code, Direction, MISSING, Mat, SequenceSet};
use transiograms_hmm::Sequence;

use crate::core::PyCategories;
use crate::{convergence, invalid};

const BAD_CODES: &str = "codes must be integers in 0..k-1 or -1 for missing";

fn asarray<'py>(obj: &Bound<'py, PyAny>, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.py()
        .import("numpy")?
        .call_method1("asarray", (obj, dtype))
}

/// 1-D float64 values of `obj`.
pub fn floats(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<f64>> {
    let array: PyReadonlyArray1<f64> = asarray(obj, "float64")
        .map_err(|_| invalid(format!("{what} must be a 1-D numeric array")))?
        .extract()
        .map_err(|_| invalid(format!("{what} must be a 1-D numeric array")))?;
    Ok(array.as_array().to_vec())
}

/// A string argument names a column of `data`; anything else is the values themselves.
pub fn column<'py>(
    data: Option<&Bound<'py, PyAny>>,
    arg: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let Ok(name) = arg.cast::<PyString>() else {
        return Ok(arg.clone());
    };
    let name = name.to_str()?;
    let data =
        data.ok_or_else(|| invalid(format!("{what} names column {name:?}; pass data= as well")))?;
    data.get_item(name).map_err(|_| {
        invalid(format!(
            "{what} names column {name:?}, which data does not have"
        ))
    })
}

/// Codes from int (-1 missing) or float (NaN missing) arrays.
pub fn codes(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<Code>> {
    floats(obj, what)
        .map_err(|_| invalid(format!("{what}: {BAD_CODES}")))?
        .into_iter()
        .map(|c| match c {
            c if c.is_nan() || c == -1.0 => Ok(MISSING),
            c if c >= 0.0 && c.fract() == 0.0 && c < MISSING as f64 => Ok(c as Code),
            c => Err(invalid(format!("{what}: {BAD_CODES}, found {c}"))),
        })
        .collect()
}

/// `k` from `categories` (an int or a Categories), else `max(code) + 1`; codes are checked.
pub fn categories_k(obj: Option<&Bound<PyAny>>, codes: &[Code]) -> PyResult<usize> {
    let k = match obj {
        None => transiograms_core::codes::n_categories(codes),
        Some(o) if o.is_none() => transiograms_core::codes::n_categories(codes),
        Some(o) => match o.cast::<PyCategories>() {
            Ok(c) => c.get().0.len(),
            Err(_) => o
                .extract::<usize>()
                .map_err(|_| invalid("categories must be an int or a Categories"))?,
        },
    };
    transiograms_core::codes::check_codes(codes, k).map_err(to_rust)?;
    Ok(k)
}

/// Per-row hole ids as text; None gives one hole "0".
pub fn hole_ids(obj: Option<&Bound<PyAny>>, n: usize) -> PyResult<Vec<String>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(vec!["0".to_string(); n]);
    };
    let list = asarray(obj, "object")?.call_method0("tolist")?;
    let list = list
        .cast::<PyList>()
        .map_err(|_| invalid("holes must be a 1-D array of hole ids"))?;
    if list.len() != n {
        return Err(invalid(format!(
            "holes has {} values for {n} rows",
            list.len()
        )));
    }
    list.iter()
        .map(|item| Ok(item.str()?.to_str()?.to_string()))
        .collect()
}

pub fn direction(s: &str) -> PyResult<Direction> {
    match s {
        "up" => Ok(Direction::Up),
        "down" => Ok(Direction::Down),
        _ => Err(invalid(format!(
            "direction must be \"up\" or \"down\", not {s:?}"
        ))),
    }
}

/// The shared sequence-data keywords (spec section 5.2).
pub struct SequenceArgs<'py> {
    pub codes: Bound<'py, PyAny>,
    pub holes: Option<Bound<'py, PyAny>>,
    pub from_: Option<Bound<'py, PyAny>>,
    pub to: Option<Bound<'py, PyAny>>,
    pub depth: Option<Bound<'py, PyAny>>,
    pub step: Option<f64>,
    pub gap: f64,
    pub categories: Option<Bound<'py, PyAny>>,
    pub data: Option<Bound<'py, PyAny>>,
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

/// Median positive within-hole spacing of `depth`.
pub fn infer_step(holes: &[String], depth: &[f64]) -> PyResult<f64> {
    let mut by_hole: std::collections::HashMap<&str, Vec<f64>> = Default::default();
    for (h, &v) in holes.iter().zip(depth) {
        by_hole.entry(h).or_default().push(v);
    }
    let mut spacings = Vec::new();
    for v in by_hole.values_mut() {
        v.sort_by(f64::total_cmp);
        spacings.extend(v.windows(2).map(|w| w[1] - w[0]).filter(|s| *s > 0.0));
    }
    median(spacings).ok_or_else(|| invalid("cannot infer step from depth; pass step="))
}

/// Intervals from `from_` and `to`, regular samples from `depth`, else unit-step codes in row
/// order per hole. Samples spaced wider than `step` leave a gap of `spacing - step`, which breaks
/// the run when larger than `gap`.
pub fn sequence_set(a: SequenceArgs) -> PyResult<SequenceSet> {
    Ok(sequence_set_step(a)?.0)
}

/// `sequence_set` and the sampling step: `step`, inferred from `depth`, 1 for unit-step codes,
/// None for intervals without `step`.
pub fn sequence_set_step(a: SequenceArgs) -> PyResult<(SequenceSet, Option<f64>)> {
    let data = a.data.as_ref();
    let col = |o: &Option<Bound<'_, PyAny>>, what: &str| -> PyResult<Option<Vec<f64>>> {
        o.as_ref()
            .filter(|o| !o.is_none())
            .map(|o| floats(&column(data, o, what)?, what))
            .transpose()
    };
    let codes = codes(&column(data, &a.codes, "codes")?, "codes")?;
    let n = codes.len();
    let k = categories_k(a.categories.as_ref(), &codes)?;
    let holes = match a.holes.as_ref().filter(|o| !o.is_none()) {
        Some(h) => hole_ids(Some(&column(data, h, "holes")?), n)?,
        None => hole_ids(None, n)?,
    };
    if a.gap.is_nan() || a.gap < 0.0 {
        return Err(invalid(format!("gap must be >= 0, got {}", a.gap)));
    }
    let (from, to, depth) = (
        col(&a.from_, "from_")?,
        col(&a.to, "to")?,
        col(&a.depth, "depth")?,
    );
    let (set, step) = match (from, to, depth) {
        (Some(_), _, Some(_)) | (_, Some(_), Some(_)) => {
            return Err(invalid("give either from_ and to, or depth, not both"));
        }
        (Some(f), Some(t), None) => (
            SequenceSet::from_intervals(&holes, &f, &t, &codes, k),
            a.step,
        ),
        (Some(_), None, None) | (None, Some(_), None) => {
            return Err(invalid("give from_ and to together"));
        }
        (None, None, Some(d)) => {
            let step = match a.step {
                Some(s) => s,
                None => infer_step(&holes, &d)?,
            };
            (
                SequenceSet::from_samples(&holes, &d, &codes, step, k),
                Some(step),
            )
        }
        (None, None, None) => {
            let mut seen: std::collections::HashMap<&str, f64> = Default::default();
            let from: Vec<f64> = holes
                .iter()
                .map(|h| {
                    let next = seen.entry(h).or_insert(0.0);
                    *next += 1.0;
                    *next - 1.0
                })
                .collect();
            let to: Vec<f64> = from.iter().map(|f| f + 1.0).collect();
            (
                SequenceSet::from_intervals(&holes, &from, &to, &codes, k),
                Some(1.0),
            )
        }
    };
    Ok((set.map_err(to_rust)?, step))
}

/// Where each step of each sequence came from, so outputs return in input row order.
#[derive(Clone, Debug, PartialEq)]
pub struct RowMap {
    /// Input row of every step, per sequence, in sequence (direction) order.
    pub rows: Vec<Vec<usize>>,
    pub n_rows: usize,
    /// Rows per input array when the input was a list of arrays.
    pub list: Option<Vec<usize>>,
    pub direction: Direction,
}

pub fn matrix(obj: &Bound<PyAny>, what: &str) -> PyResult<(usize, usize, Vec<f64>)> {
    let array =
        asarray(obj, "float64").map_err(|_| invalid(format!("{what} must be a numeric array")))?;
    let ndim: usize = array.getattr("ndim")?.extract()?;
    let array = match ndim {
        1 => array.call_method1("reshape", ((-1i64, 1i64),))?,
        2 => array,
        _ => return Err(invalid(format!("{what} must have shape (t,) or (t, d)"))),
    };
    let a: PyReadonlyArray2<f64> = array.extract()?;
    let view = a.as_array();
    let (t, d) = view.dim();
    Ok((t, d, view.iter().copied().collect()))
}

/// HMM input: a list of arrays (one per hole) or one array plus per-row `holes`.
pub fn sequences(
    obj: &Bound<PyAny>,
    holes: Option<&Bound<PyAny>>,
    data: Option<&Bound<PyAny>>,
    direction: Direction,
    d_expected: Option<usize>,
) -> PyResult<(Vec<Sequence>, RowMap)> {
    let obj = column(data, obj, "sequences")?;
    let mut pieces: Vec<(usize, Vec<f64>, Vec<usize>)> = Vec::new();
    let mut d_seen = None;
    let mut list = None;
    let mut n_rows = 0;
    let mut check_d = |d: usize| -> PyResult<()> {
        let want = d_expected.or(d_seen).unwrap_or(d);
        if d != want {
            return Err(invalid(format!(
                "every sequence must have {want} channels, found {d}"
            )));
        }
        d_seen = Some(d);
        Ok(())
    };
    if let Ok(items) = obj.cast::<PyList>() {
        if holes.is_some_and(|h| !h.is_none()) {
            return Err(invalid(
                "give holes= only with a single array, not with a list of arrays",
            ));
        }
        let mut lengths = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let (t, d, x) = matrix(&item, &format!("sequences[{i}]"))?;
            check_d(d)?;
            pieces.push((d, x, (n_rows..n_rows + t).collect()));
            lengths.push(t);
            n_rows += t;
        }
        list = Some(lengths);
    } else {
        let (t, d, x) = matrix(&obj, "sequences")?;
        check_d(d)?;
        n_rows = t;
        let ids = match holes.filter(|h| !h.is_none()) {
            Some(h) => hole_ids(Some(&column(data, h, "holes")?), t)?,
            None => hole_ids(None, t)?,
        };
        let mut order: std::collections::HashMap<&str, usize> = Default::default();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (row, id) in ids.iter().enumerate() {
            let g = *order.entry(id).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(row);
        }
        for rows in groups {
            let values = rows
                .iter()
                .flat_map(|&r| x[r * d..(r + 1) * d].iter().copied())
                .collect();
            pieces.push((d, values, rows));
        }
    }
    let mut seqs = Vec::with_capacity(pieces.len());
    let mut maps = Vec::with_capacity(pieces.len());
    for (d, mut x, mut rows) in pieces {
        if direction == Direction::Up {
            rows.reverse();
            x = x.chunks(d.max(1)).rev().flatten().copied().collect();
        }
        seqs.push(Sequence {
            t: rows.len(),
            d,
            x,
        });
        maps.push(rows);
    }
    Ok((
        seqs,
        RowMap {
            rows: maps,
            n_rows,
            list,
            direction,
        },
    ))
}

/// `(n, 3)` coordinates; `(n, 2)` gets z = 0.
pub fn coords(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let (n, d, x) = matrix(obj, what)?;
    match d {
        3 => Ok((0..n)
            .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
            .collect()),
        2 => Ok((0..n).map(|i| [x[2 * i], x[2 * i + 1], 0.0]).collect()),
        _ => Err(invalid(format!("{what} must have shape (n, 3) or (n, 2)"))),
    }
}

pub fn to_rust(e: transiograms_core::Error) -> PyErr {
    match e {
        transiograms_core::Error::InvalidInput(m) => invalid(m),
        transiograms_core::Error::Convergence(m) => convergence(m),
    }
}

pub fn vec1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, v)
}

pub fn mat2<'py>(py: Python<'py>, m: &Mat) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)]).into_pyarray(py)
}

/// `(n, r, c)` stack of equally shaped matrices.
pub fn mat3<'py>(py: Python<'py>, ms: &[Mat], r: usize, c: usize) -> Bound<'py, PyArray3<f64>> {
    Array3::from_shape_fn((ms.len(), r, c), |(n, i, j)| ms[n][(i, j)]).into_pyarray(py)
}

/// A 2-D float64 matrix.
pub fn mat(obj: &Bound<PyAny>, what: &str) -> PyResult<Mat> {
    let array =
        asarray(obj, "float64").map_err(|_| invalid(format!("{what} must be a numeric matrix")))?;
    let a: PyReadonlyArray2<f64> = array
        .extract()
        .map_err(|_| invalid(format!("{what} must be a 2-D numeric matrix")))?;
    let v = a.as_array();
    let (r, c) = v.dim();
    Ok(Mat::from_fn(r, c, |i, j| v[(i, j)]))
}

/// `(n, 3)` lags or one `(3,)` lag; the flag says whether one lag was given.
pub fn lags3(obj: &Bound<PyAny>, what: &str) -> PyResult<(Vec<[f64; 3]>, bool)> {
    let array = asarray(obj, "float64").map_err(|_| invalid(format!("{what} must be numeric")))?;
    let ndim: usize = array.getattr("ndim")?.extract()?;
    let single = ndim == 1;
    let (n, d, x) = matrix(&array, what)?;
    let (n, d, x) = if single { (1, n, x) } else { (n, d, x) };
    if d != 3 {
        return Err(invalid(format!("{what} must have shape (3,) or (n, 3)")));
    }
    Ok((
        (0..n)
            .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
            .collect(),
        single,
    ))
}

/// Whether `obj` is 0-dimensional; size-1 arrays are not scalars on any NumPy version.
pub fn is_scalar(obj: &Bound<PyAny>) -> PyResult<bool> {
    Ok(obj
        .py()
        .import("numpy")?
        .call_method1("ndim", (obj,))?
        .extract::<usize>()?
        == 0)
}

/// A float or a 1-D array of floats; the flag says whether a scalar was given.
pub fn scalar_or_floats(obj: &Bound<PyAny>, what: &str) -> PyResult<(Vec<f64>, bool)> {
    if is_scalar(obj)? {
        let v = obj
            .extract::<f64>()
            .map_err(|_| invalid(format!("{what} must be a number or a 1-D numeric array")))?;
        return Ok((vec![v], true));
    }
    Ok((floats(obj, what)?, false))
}

/// Hole ids as u32 indices in first-appearance order.
pub fn hole_indices(ids: &[String]) -> Vec<u32> {
    let mut seen: std::collections::HashMap<&str, u32> = Default::default();
    ids.iter()
        .map(|h| {
            let n = seen.len() as u32;
            *seen.entry(h).or_insert(n)
        })
        .collect()
}

/// Mask of `n` booleans, or None.
pub fn mask(obj: Option<&Bound<PyAny>>, n: usize) -> PyResult<Option<Vec<bool>>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    let m: Vec<bool> = asarray(obj, "bool")?
        .call_method1("reshape", (-1i64,))?
        .call_method0("tolist")?
        .extract()
        .map_err(|_| invalid("mask must be a boolean array"))?;
    if m.len() != n {
        return Err(invalid(format!(
            "mask has {} values for {n} cells",
            m.len()
        )));
    }
    Ok(Some(m))
}
