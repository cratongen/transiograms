//! `Prediction`, `Realizations`, `tp_cokriging` and `tprogs_simulate`.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use transiograms_core::{Grid, MISSING, Prediction, Realizations, Snapped, memory};
use transiograms_tpsim::{Options, QuenchOptions};
use transiograms_transiogram::TransitionModel;

use crate::args::{codes, column, coords, mask, to_rust, vec1};
use crate::core::{PyGrid, json, same_json, unjson};
use crate::invalid;
use crate::transiogram::PyTransitionModel;

/// Hard data snapped to the grid and the mask, checked against the model.
pub(crate) fn hard<'py>(
    model: &TransitionModel,
    grid: &Grid,
    xyz: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Snapped, Option<Vec<bool>>)> {
    let xyz = self::coords(&column(data, xyz, "coords")?, "coords")?;
    let c = self::codes(&column(data, codes, "codes")?, "codes")?;
    if xyz.len() != c.len() {
        return Err(invalid(format!(
            "{} coordinates but {} codes",
            xyz.len(),
            c.len()
        )));
    }
    let k = model.k();
    if let Some(&bad) = c.iter().find(|&&c| c != MISSING && c as usize >= k) {
        return Err(invalid(format!(
            "code {bad} is not one of the model's {k} categories"
        )));
    }
    let m = self::mask(mask, grid.n_cells())?;
    let s = grid.snap(&xyz, &c, m.as_deref()).map_err(to_rust)?;
    Ok((s, m))
}

fn probabilities<'py>(py: Python<'py>, k: usize, p: &[f64]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray1::from_slice(py, p).reshape([p.len() / k.max(1), k])
}

fn valid_codes(c: &[u32], k: usize, what: &str) -> PyResult<()> {
    match c.iter().find(|&&c| c != MISSING && c as usize >= k) {
        Some(bad) => Err(invalid(format!(
            "{what} holds code {bad}, not one of {k} categories"
        ))),
        None => Ok(()),
    }
}

fn codes_i64<'py>(py: Python<'py>, c: &[u32]) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(
        py,
        c.iter()
            .map(|&c| if c == MISSING { -1 } else { c as i64 })
            .collect(),
    )
}

/// Category probabilities at every cell.
///
/// Attributes
/// ----------
/// probabilities : numpy.ndarray
///     ``(n_cells, k)``, rows summing to 1; NaN outside the mask.
/// most_likely : numpy.ndarray
///     int64 most probable category (lowest code on ties), -1 outside the mask.
/// entropy : numpy.ndarray
///     Shannon entropy over ``ln k``, in [0, 1].
/// corrected : float
///     Fraction of cells whose probabilities were clipped to [0, 1] and renormalised.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> pred = tg.tp_cokriging(model, grid, xyz, codes)
/// >>> pred.probabilities.shape, pred.entropy.shape
/// ((864, 3), (864,))
/// >>> pred.most_likely[cells], hard
/// (array([0, 2, 1]), array([0, 2, 1]))
/// >>> float(pred.corrected) >= 0.0
/// True
/// >>> tg.Prediction.from_json(pred.to_json()).to_json() == pred.to_json()
/// True
#[pyclass(module = "transiograms", name = "Prediction", frozen)]
pub struct PyPrediction(pub Prediction);

#[pymethods]
impl PyPrediction {
    /// ``(n_cells, k)`` category probabilities, rows summing to 1; NaN outside the mask.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        probabilities(py, self.0.k, &self.0.probabilities)
    }
    /// int64 most probable category (lowest code on ties), -1 outside the mask.
    #[getter]
    fn most_likely<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        codes_i64(py, &self.0.most_likely)
    }
    /// Shannon entropy of ``probabilities`` over ``ln k``, in [0, 1].
    #[getter]
    fn entropy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.entropy.clone())
    }
    /// Fraction of cells whose probabilities were clipped to [0, 1] and renormalised.
    #[getter]
    fn corrected(&self) -> f64 {
        self.0.corrected
    }
    /// JSON text of this object; ``Prediction.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// Prediction from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Prediction
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not Prediction JSON or describes an invalid Prediction.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let p: Prediction = unjson(text, "Prediction")?;
        let cells = p.most_likely.len();
        if p.k == 0 || p.probabilities.len() != cells * p.k || p.entropy.len() != cells {
            return Err(invalid(format!(
                "Prediction JSON with {cells} cells needs k >= 1, {cells} * k probabilities and \
                 {cells} entropies"
            )));
        }
        valid_codes(&p.most_likely, p.k, "most_likely")?;
        Ok(Self(p))
    }
    fn __eq__(&self, other: PyRef<Self>) -> PyResult<bool> {
        same_json(&self.0, &other.0)
    }
    fn __repr__(&self) -> String {
        format!(
            "Prediction(n_cells={}, k={}, corrected={})",
            self.0.most_likely.len(),
            self.0.k,
            self.0.corrected
        )
    }
}

/// Categorical grid realizations.
///
/// Attributes
/// ----------
/// realizations : numpy.ndarray
///     int64 ``(n, n_cells)``, cells x fastest, -1 outside the mask.
/// proportions : numpy.ndarray
///     ``(n, k)`` category proportions of each realization over active cells.
/// probabilities : numpy.ndarray
///     ``(n_cells, k)`` frequency of each category over the realizations.
/// most_likely, entropy : numpy.ndarray
///     Of ``probabilities``, as in ``Prediction``.
/// corrected : numpy.ndarray
///     ``(n,)`` fraction of cells clipped during sequential simulation (zeros for MCRF).
/// objective : list of numpy.ndarray
///     Quenching objective after each sweep, per realization (empty for MCRF).
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> real = tg.mcrf_simulate(model, grid, xyz, codes, n=3, seed=1)
/// >>> real.realizations.shape, real.proportions.shape, real.probabilities.shape
/// ((3, 864), (3, 3), (864, 3))
/// >>> bool(np.all(real.realizations[:, cells] == hard))
/// True
/// >>> real.most_likely.shape, real.entropy.shape, real.corrected, real.objective
/// ((864,), (864,), array([0., 0., 0.]), [])
/// >>> tg.Realizations.from_json(real.to_json()).to_json() == real.to_json()
/// True
#[pyclass(module = "transiograms", name = "Realizations", frozen)]
pub struct PyRealizations {
    pub r: Realizations,
    pub memory: Option<u64>,
}

impl PyRealizations {
    fn frequencies(&self) -> PyResult<Prediction> {
        let (n, cells, k) = (self.r.n, self.r.n_cells, self.r.k);
        memory::check(
            (cells as u64).saturating_mul(k as u64 * 8 + 12),
            self.memory,
            "realization frequencies",
        )
        .map_err(to_rust)?;
        let mut p = vec![0.0; cells * k];
        for r in 0..n {
            for (c, &code) in self.r.codes[r * cells..(r + 1) * cells].iter().enumerate() {
                if code != MISSING {
                    p[c * k + code as usize] += 1.0;
                }
            }
        }
        for row in p.chunks_mut(k) {
            let total: f64 = row.iter().sum();
            for v in row {
                *v = if total > 0.0 { *v / total } else { f64::NAN };
            }
        }
        Ok(Prediction::from_probabilities(k, p, 0.0))
    }
}

#[pymethods]
impl PyRealizations {
    /// int64 ``(n, n_cells)``, cells x fastest, -1 outside the mask.
    #[getter]
    fn realizations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let (n, cells) = (self.r.n, self.r.n_cells);
        memory::check(
            (n as u64).saturating_mul(cells as u64 * 8),
            self.memory,
            "int64 realizations",
        )
        .map_err(to_rust)?;
        codes_i64(py, &self.r.codes).reshape([n, cells])
    }
    /// ``(n, k)`` category proportions of each realization over active cells.
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, cells, k) = (self.r.n, self.r.n_cells, self.r.k);
        let mut p = vec![0.0; n * k];
        for r in 0..n {
            let row = &self.r.codes[r * cells..(r + 1) * cells];
            let active = row.iter().filter(|&&c| c != MISSING).count().max(1) as f64;
            for &c in row.iter().filter(|&&c| c != MISSING) {
                p[r * k + c as usize] += 1.0 / active;
            }
        }
        probabilities(py, k, &p)
    }
    /// ``(n_cells, k)`` frequency of each category over the realizations.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        probabilities(py, self.r.k, &self.frequencies()?.probabilities)
    }
    /// int64 most frequent category per cell (lowest code on ties), -1 outside the mask.
    #[getter]
    fn most_likely<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(codes_i64(py, &self.frequencies()?.most_likely))
    }
    /// Shannon entropy of ``probabilities`` over ``ln k``, in [0, 1].
    #[getter]
    fn entropy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(py, self.frequencies()?.entropy))
    }
    /// ``(n,)`` fraction of cells clipped during sequential simulation (zeros for MCRF).
    #[getter]
    fn corrected<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.r.corrected.clone())
    }
    /// Quenching objective after each sweep, per realization (empty for MCRF).
    #[getter]
    fn objective<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.r.objective.iter().map(|o| vec1(py, o.clone())))
    }
    fn __len__(&self) -> usize {
        self.r.n
    }
    /// JSON text of this object; ``Realizations.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        memory::check(
            (self.r.codes.len() as u64).saturating_mul(12),
            self.memory,
            "Realizations JSON",
        )
        .map_err(to_rust)?;
        json(&self.r)
    }
    /// Realizations from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Realizations
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not Realizations JSON or describes an invalid Realizations.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let r: Realizations = unjson(text, "Realizations")?;
        if r.k == 0 || r.n.checked_mul(r.n_cells) != Some(r.codes.len()) || r.corrected.len() != r.n
        {
            return Err(invalid(format!(
                "Realizations JSON needs k >= 1, n * n_cells = {} * {} codes and n corrected values",
                r.n, r.n_cells
            )));
        }
        valid_codes(&r.codes, r.k, "codes")?;
        Ok(Self { r, memory: None })
    }
    fn __eq__(&self, other: PyRef<Self>) -> PyResult<bool> {
        same_json(&self.r, &other.r)
    }
    fn __repr__(&self) -> String {
        format!(
            "Realizations(n={}, n_cells={}, k={})",
            self.r.n, self.r.n_cells, self.r.k
        )
    }
}

/// Transition-probability cokriging (T-PROGS): category probabilities at every cell from the
/// hard data, without simulation.
///
/// Parameters
/// ----------
/// model : TransitionModel
/// grid : Grid
/// coords : array_like
///     ``(n, 3)`` hard data coordinates (composite midpoints).
/// codes : array_like or str
///     Hard data categories.
/// mask : array_like, optional
///     Active cells.
/// max_data : int
///     Nearest data per cell, even.
/// closeness_limit : float
///     Offsets whose closeness falls below it are not searched.
/// max_per_octant : int
///     0 for no cap.
/// svd_cutoff : float
///     Relative singular value cutoff of the cokriging solve.
/// memory : int, optional
///     Byte budget; the available memory by default.
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
/// ConvergenceError
///     When a cokriging system cannot be solved.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> pred = tg.tp_cokriging(model, grid, xyz, codes, max_data=8)
/// >>> bool(np.allclose(pred.probabilities.sum(axis=1), 1.0))
/// True
#[pyfunction]
#[pyo3(signature = (model, grid, coords, codes, *, mask=None, max_data=12, closeness_limit=0.01, max_per_octant=0, svd_cutoff=1e-8, memory=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn tp_cokriging<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &PyGrid,
    coords: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    max_data: usize,
    closeness_limit: f64,
    max_per_octant: usize,
    svd_cutoff: f64,
    memory: Option<u64>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyPrediction> {
    let (h, m) = hard(&model.0, &grid.0, coords, codes, mask, data)?;
    let opts = Options {
        max_data,
        closeness_limit,
        max_per_octant,
        svd_cutoff,
        quench: None,
        memory,
        ..Options::default()
    };
    py.detach(|| transiograms_tpsim::predict(&model.0, &grid.0, m.as_deref(), &h, &opts))
        .map(PyPrediction)
        .map_err(to_rust)
}

/// T-PROGS simulation: sequential indicator simulation with transition-probability cokriging,
/// then zero-temperature quenching towards the model transiograms. Hard cells never change.
///
/// Parameters
/// ----------
/// model, grid, coords, codes, mask, max_data, closeness_limit, max_per_octant, svd_cutoff
///     As in ``tp_cokriging``.
/// n : int
///     Number of realizations.
/// seed : int
/// quench : bool
/// quench_iterations : int
///     Largest number of quenching sweeps.
/// quench_limit : float
///     Closeness above which a lag enters the quenching objective.
/// quench_tol : float
///     Relative objective gain that stops quenching.
/// proportion_weight : float
///     Weight of the proportion term in the objective.
/// memory : int, optional
/// data : mapping, optional
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
/// ConvergenceError
///     When a cokriging system cannot be solved.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> real = tg.tprogs_simulate(model, grid, xyz, codes, n=2, seed=1, quench_iterations=1)
/// >>> real.realizations.shape, len(real.objective)
/// ((2, 864), 2)
/// >>> bool(np.all(real.realizations[:, cells] == hard))
/// True
#[pyfunction]
#[pyo3(signature = (model, grid, coords, codes, *, n=1, seed=0, mask=None, max_data=12, closeness_limit=0.01, max_per_octant=0, svd_cutoff=1e-8, quench=true, quench_iterations=3, quench_limit=0.4, quench_tol=1e-3, proportion_weight=1.0, memory=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn tprogs_simulate<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &PyGrid,
    coords: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    n: usize,
    seed: i128,
    mask: Option<&Bound<'py, PyAny>>,
    max_data: usize,
    closeness_limit: f64,
    max_per_octant: usize,
    svd_cutoff: f64,
    quench: bool,
    quench_iterations: usize,
    quench_limit: f64,
    quench_tol: f64,
    proportion_weight: f64,
    memory: Option<u64>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyRealizations> {
    let seed = crate::args::seed(seed)?;
    let (h, m) = hard(&model.0, &grid.0, coords, codes, mask, data)?;
    let opts = Options {
        n_realizations: n,
        seed,
        max_data,
        closeness_limit,
        max_per_octant,
        svd_cutoff,
        quench: quench.then_some(QuenchOptions {
            max_iter: quench_iterations,
            limit: quench_limit,
            tol: quench_tol,
            proportion_weight,
        }),
        memory,
    };
    let r = py
        .detach(|| transiograms_tpsim::simulate(&model.0, &grid.0, m.as_deref(), &h, &opts))
        .map_err(to_rust)?;
    Ok(PyRealizations { r, memory })
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPrediction>()?;
    m.add_class::<PyRealizations>()?;
    m.add_function(wrap_pyfunction!(tp_cokriging, m)?)?;
    m.add_function(wrap_pyfunction!(tprogs_simulate, m)?)?;
    Ok(())
}
