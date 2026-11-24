//! `mcp_predict` and `mcrf_simulate`.

use pyo3::prelude::*;
use transiograms_mcrf::{Neighbourhood, Options};

use crate::args::to_rust;
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
/// neighbours : "sectors", int or None
///     Nearest datum per quadrant (2D) or octant (3D) ("sectors" or None), or the nearest
///     ``neighbours`` data.
/// closeness_limit : float
/// memory : int, optional
/// data : mapping, optional
///
/// Returns
/// -------
/// Prediction
///
/// Raises
/// ------
/// InvalidInput
///     When the coordinates are not ``(n, 3)``, the lengths differ, a code is not a category
///     of ``model``, ``mask`` has another length than the grid, an option is out of range,
///     ``seed`` is out of range, or the work would need more than ``memory`` (the available
///     memory by default); the message names the bytes.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> pred = tg.mcp_predict(model, grid, xyz, codes)
/// >>> pred.probabilities.shape
/// (864, 3)
/// >>> pred.most_likely[cells], hard
/// (array([0, 2, 1]), array([0, 2, 1]))
#[pyfunction]
#[pyo3(signature = (model, grid, coords, codes=None, *, mask=None, neighbours=None, closeness_limit=0.01, memory=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn mcp_predict<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    codes: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    neighbours: Option<&Bound<'py, PyAny>>,
    closeness_limit: f64,
    memory: Option<u64>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyPrediction> {
    let h = hard(&model.0, grid, coords, codes, mask, data)?;
    let opts = options(neighbours, closeness_limit, 1, 0, memory)?;
    py.detach(|| {
        transiograms_mcrf::predict(&model.0, &h.grid, h.mask.as_deref(), &h.snapped, &opts)
    })
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
///
/// Raises
/// ------
/// InvalidInput
///     When the coordinates are not ``(n, 3)``, the lengths differ, a code is not a category
///     of ``model``, ``mask`` has another length than the grid, an option is out of range,
///     ``seed`` is out of range, or the work would need more than ``memory`` (the available
///     memory by default); the message names the bytes.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> real = tg.mcrf_simulate(model, grid, xyz, codes, n=2, seed=1, neighbours=8)
/// >>> real.realizations.shape
/// (2, 864)
/// >>> bool(np.array_equal(real.realizations, tg.mcrf_simulate(model, grid, xyz, codes, n=2, seed=1, neighbours=8).realizations))
/// True
#[pyfunction]
#[pyo3(signature = (model, grid, coords, codes=None, *, n=1, seed=0, mask=None, neighbours=None, closeness_limit=0.01, memory=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn mcrf_simulate<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    codes: Option<&Bound<'py, PyAny>>,
    n: usize,
    seed: i128,
    mask: Option<&Bound<'py, PyAny>>,
    neighbours: Option<&Bound<'py, PyAny>>,
    closeness_limit: f64,
    memory: Option<u64>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyRealizations> {
    let seed = crate::args::seed(seed)?;
    let h = hard(&model.0, grid, coords, codes, mask, data)?;
    let opts = options(neighbours, closeness_limit, n, seed, memory)?;
    let r = py
        .detach(|| {
            transiograms_mcrf::simulate(&model.0, &h.grid, h.mask.as_deref(), &h.snapped, &opts)
        })
        .map_err(to_rust)?;
    Ok(PyRealizations {
        r,
        memory,
        grid: h.grid,
        categories: h.categories,
        crs: h.crs,
        length_unit: h.length_unit,
    })
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcp_predict, m)?)?;
    m.add_function(wrap_pyfunction!(mcrf_simulate, m)?)?;
    Ok(())
}
