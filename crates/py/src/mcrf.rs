//! `mcp_predict` and `mcrf_simulate`.

use pyo3::prelude::*;
use transiograms_mcrf::{Neighbourhood, Options};

use crate::args::to_rust;
use crate::core::PyGrid;
use crate::invalid;
use crate::tpsim::{PyPrediction, PyRealizations, hard};
use crate::transiogram::PyTransitionModel;

fn neighbourhood(obj: &Bound<PyAny>) -> PyResult<Neighbourhood> {
    if let Ok(s) = obj.extract::<String>() {
        return match s.as_str() {
            "sectors" => Ok(Neighbourhood::Sectors),
            _ => Err(invalid(format!(
                "neighbours must be \"sectors\" or a count, not {s:?}"
            ))),
        };
    }
    match obj.extract::<usize>() {
        Ok(q) if q > 0 => Ok(Neighbourhood::Nearest(q)),
        _ => Err(invalid(
            "neighbours must be \"sectors\" or a positive count",
        )),
    }
}

fn options(
    neighbours: Option<&Bound<PyAny>>,
    closeness_limit: f64,
    n: usize,
    seed: u64,
    memory: Option<u64>,
) -> PyResult<Options> {
    Ok(Options {
        neighbourhood: match neighbours.filter(|o| !o.is_none()) {
            Some(o) => neighbourhood(o)?,
            None => Neighbourhood::Sectors,
        },
        closeness_limit,
        n_realizations: n,
        seed,
        memory,
    })
}

/// Markov-type categorical prediction (Li and Zhang): at each cell, the probability of each
/// category from its nearest hard data, ``p_k prod_g t_{k, c_g}(h_g)`` normalised.
///
/// Parameters
/// ----------
/// model : TransitionModel
/// grid : Grid
/// coords : array_like
///     ``(n, 3)`` hard data coordinates.
/// codes : array_like or str
/// mask : array_like, optional
/// neighbours : "sectors" or int
///     Nearest datum per quadrant (2D) or octant (3D), or the nearest ``neighbours`` data.
/// closeness_limit : float
/// memory : int, optional
/// data : mapping, optional
///
/// Returns
/// -------
/// Prediction
#[pyfunction]
#[pyo3(signature = (model, grid, coords, codes, *, mask=None, neighbours=None, closeness_limit=0.01, memory=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn mcp_predict<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &PyGrid,
    coords: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    neighbours: Option<&Bound<'py, PyAny>>,
    closeness_limit: f64,
    memory: Option<u64>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyPrediction> {
    let (h, m) = hard(&model.0, &grid.0, coords, codes, mask, data)?;
    let opts = options(neighbours, closeness_limit, 1, 0, memory)?;
    py.detach(|| transiograms_mcrf::predict(&model.0, &grid.0, m.as_deref(), &h, &opts))
        .map(PyPrediction)
        .map_err(to_rust)
}

/// Sequential Markov chain random field simulation: cells visited along a seeded random path,
/// each drawn from its MCP conditional given the hard data and the cells already simulated.
///
/// Parameters
/// ----------
/// model, grid, coords, codes, mask, neighbours, closeness_limit, memory, data
///     As in ``mcp_predict``.
/// n : int
///     Number of realizations.
/// seed : int
///
/// Returns
/// -------
/// Realizations
#[pyfunction]
#[pyo3(signature = (model, grid, coords, codes, *, n=1, seed=0, mask=None, neighbours=None, closeness_limit=0.01, memory=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn mcrf_simulate<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &PyGrid,
    coords: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    n: usize,
    seed: i128,
    mask: Option<&Bound<'py, PyAny>>,
    neighbours: Option<&Bound<'py, PyAny>>,
    closeness_limit: f64,
    memory: Option<u64>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyRealizations> {
    let seed = crate::args::seed(seed)?;
    let (h, m) = hard(&model.0, &grid.0, coords, codes, mask, data)?;
    let opts = options(neighbours, closeness_limit, n, seed, memory)?;
    let r = py
        .detach(|| transiograms_mcrf::simulate(&model.0, &grid.0, m.as_deref(), &h, &opts))
        .map_err(to_rust)?;
    Ok(PyRealizations { r, memory })
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcp_predict, m)?)?;
    m.add_function(wrap_pyfunction!(mcrf_simulate, m)?)?;
    Ok(())
}
