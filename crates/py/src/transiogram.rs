//! Experimental transiograms, `TransiogramModel` (1D) and `TransitionModel` (3D).

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use transiograms_core::{Mat, RateMatrix, memory};
use transiograms_transiogram::{
    Entry, Experimental, FitQuality, Model1D, PairFilter, PairSpec, Structure, TransitionModel,
    downhole, experimental, fit_quality, on_grid,
};

use crate::args::{
    SequenceArgs, categories_k, codes, column, coords, direction, floats, hole_ids, hole_indices,
    lags3, mask, mat, mat2, mat3, scalar_or_floats, sequence_set, to_rust, vec1,
};
use crate::chain::{PyMarkovChain, background_index};
use crate::core::{PyGrid, json, unjson};
use crate::invalid;
use crate::tpsim::PyRealizations;

/// Experimental transition probabilities per lag.
///
/// Attributes
/// ----------
/// lags : numpy.ndarray
///     Nominal lag of each class.
/// mean_lags : numpy.ndarray
///     Mean separation of the pairs in each class (NaN when empty).
/// counts : numpy.ndarray
///     ``(n_lags, k, k)`` pair counts (lengths for the downhole estimator).
/// probabilities : numpy.ndarray
///     ``(n_lags, k, k)`` counts over row totals; NaN rows where a tail category has no pair.
/// tail_totals : numpy.ndarray
///     ``(n_lags, k)`` row totals.
/// directed : bool
///     False when both orientations of each pair were counted.
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> exp = tg.downhole_transiogram(
/// ...     d["CODE"], lags=[0.5, 1.0, 1.5, 2.0], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> exp.lags
/// array([0.5, 1. , 1.5, 2. ])
/// >>> exp.counts.shape, exp.probabilities.shape, exp.tail_totals.shape
/// ((4, 3, 3), (4, 3, 3), (4, 3))
/// >>> bool(np.allclose(exp.probabilities.sum(axis=2), 1.0))
/// True
/// >>> tg.ExperimentalTransiogram.from_json(exp.to_json()).to_json() == exp.to_json()
/// True
#[pyclass(
    module = "transiograms",
    name = "ExperimentalTransiogram",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyExperimental(pub Experimental);

impl PyExperimental {
    fn k(&self) -> usize {
        self.0.counts.first().map_or(0, |m| m.nrows())
    }
}

#[pymethods]
impl PyExperimental {
    /// ``(n_lags,)`` nominal lag of each class.
    #[getter]
    fn lags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.lags.clone())
    }
    /// ``(n_lags,)`` mean separation of the pairs in each class, NaN when empty.
    #[getter]
    fn mean_lags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.mean_lags.clone())
    }
    /// ``(n_lags, k, k)`` pair counts (lengths for the downhole estimator).
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        mat3(py, &self.0.counts, self.k(), self.k())
    }
    /// ``(n_lags, k, k)`` counts over row totals; NaN rows where a tail category has no pair.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        mat3(py, &self.0.probabilities, self.k(), self.k())
    }
    /// ``(n_lags, k)`` row totals.
    #[getter]
    fn tail_totals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let t = &self.0.tail_totals;
        mat2(py, &Mat::from_fn(t.len(), self.k(), |l, i| t[l][i]))
    }
    /// False when both orientations of each pair were counted.
    #[getter]
    fn directed(&self) -> bool {
        self.0.directed
    }
    /// JSON text of this object; ``ExperimentalTransiogram.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// ExperimentalTransiogram from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// ExperimentalTransiogram
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not ExperimentalTransiogram JSON or describes an invalid ExperimentalTransiogram.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        unjson(text, "ExperimentalTransiogram").map(Self)
    }
    fn __repr__(&self) -> String {
        format!(
            "ExperimentalTransiogram(n_lags={}, n_categories={})",
            self.0.lags.len(),
            self.k()
        )
    }
}

/// Transiogram from scattered points (composites).
///
/// Parameters
/// ----------
/// coords : array_like
///     ``(n, 3)`` or ``(n, 2)`` coordinates.
/// codes : array_like or str
///     Category per point, -1 or NaN missing.
/// holes : array_like or str, optional
///     Hole id per point, needed by ``pairs`` other than "all".
/// direction : tuple of (azimuth, dip), optional
///     Degrees; None is omnidirectional.
/// lag, n_lags : float, int
///     Lag spacing and number of lag classes.
/// lag_tolerance : float, optional
///     Half-width of a lag class; ``lag / 2`` by default.
/// angle_tolerance : float
///     Degrees.
/// bandwidth, vertical_bandwidth : float
///     Largest horizontal and vertical offset from the direction line.
/// directed : bool
///     Count pairs only head ahead of tail along the direction.
/// pairs : {"all", "same_hole", "different_hole"}
/// categories : Categories or int, optional
/// data : mapping, optional
///     Strings name its columns.
///
/// Returns
/// -------
/// ExperimentalTransiogram
///
/// Raises
/// ------
/// InvalidInput
///     When ``coords`` is not ``(n, 2)`` or ``(n, 3)``, the lengths differ, ``lag`` or
///     ``n_lags`` is not positive, ``pairs`` needs ``holes``, or an option is unknown.
///
/// Examples
/// --------
/// >>> codes = tg.MarkovChain.from_matrix([[0.8, 0.2], [0.3, 0.7]]).simulate(200, seed=1)[0]
/// >>> xyz = np.column_stack([np.zeros(200), np.zeros(200), np.arange(200.0)])
/// >>> exp = tg.experimental_transiogram(xyz, codes, lag=1.0, n_lags=3)
/// >>> exp.probabilities.shape, exp.directed
/// ((3, 2, 2), False)
#[pyfunction]
#[pyo3(signature = (coords, codes, *, holes=None, direction=None, lag, n_lags, lag_tolerance=None, angle_tolerance=22.5, bandwidth=f64::INFINITY, vertical_bandwidth=f64::INFINITY, directed=true, pairs="all", categories=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn experimental_transiogram<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    holes: Option<&Bound<'py, PyAny>>,
    direction: Option<(f64, f64)>,
    lag: f64,
    n_lags: usize,
    lag_tolerance: Option<f64>,
    angle_tolerance: f64,
    bandwidth: f64,
    vertical_bandwidth: f64,
    directed: bool,
    pairs: &str,
    categories: Option<&Bound<'py, PyAny>>,
    data: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyExperimental> {
    let xyz = self::coords(coords, "coords")?;
    let c = self::codes(&column(data, codes, "codes")?, "codes")?;
    let k = categories_k(categories, &c)?;
    let holes = holes
        .filter(|o| !o.is_none())
        .map(|h| -> PyResult<Vec<u32>> {
            Ok(hole_indices(&hole_ids(
                Some(&column(data, h, "holes")?),
                c.len(),
            )?))
        })
        .transpose()?;
    let pairs = match pairs {
        "all" => PairFilter::All,
        "same_hole" => PairFilter::SameHole,
        "different_hole" => PairFilter::DifferentHole,
        p => {
            return Err(invalid(format!(
                "pairs must be \"all\", \"same_hole\" or \"different_hole\", not {p:?}"
            )));
        }
    };
    let spec = PairSpec {
        direction: direction.map(|(a, d)| [a, d]),
        lag,
        n_lags,
        lag_tolerance: lag_tolerance.unwrap_or(lag / 2.0),
        angle_tolerance,
        bandwidth,
        vertical_bandwidth,
        directed,
        pairs,
    };
    py.detach(|| experimental(&xyz, &c, holes.as_deref(), k, &spec))
        .map(PyExperimental)
        .map_err(to_rust)
}

/// Exact downhole transiogram from intervals or samples: overlap lengths at each lag along
/// the holes.
///
/// Parameters
/// ----------
/// codes : array_like or str
/// lags : array_like
///     Positive lags.
/// direction : {"up", "down"}
/// holes, from_, to, depth, step, gap, categories, data
///     The shared sequence arguments.
///
/// Returns
/// -------
/// ExperimentalTransiogram
///
/// Raises
/// ------
/// InvalidInput
///     When a lag is not positive and finite, a code is out of range, or the sequence
///     arguments disagree in length.
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> exp = tg.downhole_transiogram(
/// ...     d["CODE"], lags=[0.5, 1.0, 1.5, 2.0], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> exp.probabilities.shape
/// (4, 3, 3)
#[pyfunction]
#[pyo3(signature = (codes, *, lags, direction="up", holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn downhole_transiogram<'py>(
    py: Python<'py>,
    codes: Bound<'py, PyAny>,
    lags: &Bound<'py, PyAny>,
    direction: &str,
    holes: Option<Bound<'py, PyAny>>,
    from_: Option<Bound<'py, PyAny>>,
    to: Option<Bound<'py, PyAny>>,
    depth: Option<Bound<'py, PyAny>>,
    step: Option<f64>,
    gap: f64,
    categories: Option<Bound<'py, PyAny>>,
    data: Option<Bound<'py, PyAny>>,
) -> PyResult<PyExperimental> {
    let lags = floats(lags, "lags")?;
    let dir = self::direction(direction)?;
    let seqs = sequence_set(SequenceArgs {
        codes,
        holes,
        from_,
        to,
        depth,
        step,
        gap,
        categories,
        data,
    })?;
    py.detach(|| downhole(&seqs, dir, &lags, gap))
        .map(PyExperimental)
        .map_err(to_rust)
}

/// Transiogram of a categorical grid along one cell offset.
///
/// Parameters
/// ----------
/// codes : array_like
///     One code per cell, x fastest; -1 for inactive cells.
/// grid : Grid
/// offset : tuple of 3 int
///     Cell step of lag 1, for example ``(0, 0, 1)`` for +z.
/// n_lags : int
/// mask : array_like, optional
///     Active cells.
/// categories : Categories or int, optional
///
/// Returns
/// -------
/// ExperimentalTransiogram
///
/// Raises
/// ------
/// InvalidInput
///     When ``codes`` or ``mask`` has another length than the grid, a code is out of range,
///     or ``offset`` is zero.
///
/// Examples
/// --------
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 1, 6))
/// >>> codes = np.repeat([0, 0, 1, 1, 2, 2], 4)
/// >>> exp = tg.grid_transiogram(codes, grid, offset=(0, 0, 1), n_lags=2)
/// >>> exp.probabilities[0]
/// array([[0.5, 0.5, 0. ],
///        [0. , 0.5, 0.5],
///        [0. , 0. , 1. ]])
#[pyfunction]
#[pyo3(signature = (codes, grid, *, offset, n_lags, mask=None, categories=None))]
fn grid_transiogram<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    grid: &PyGrid,
    offset: [i64; 3],
    n_lags: usize,
    mask: Option<&Bound<'py, PyAny>>,
    categories: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyExperimental> {
    let c = self::codes(codes, "codes")?;
    let k = categories_k(categories, &c)?;
    let m = self::mask(mask, grid.0.n_cells())?;
    py.detach(|| on_grid(&c, &grid.0, m.as_deref(), offset, n_lags, k))
        .map(PyExperimental)
        .map_err(to_rust)
}

fn structure(s: &str) -> PyResult<Structure> {
    match s {
        "exponential" => Ok(Structure::Exponential),
        "spherical" => Ok(Structure::Spherical),
        "gaussian" => Ok(Structure::Gaussian),
        _ => Err(invalid(format!(
            "structures must be \"exponential\", \"spherical\" or \"gaussian\", not {s:?}"
        ))),
    }
}

fn structures(obj: &Bound<PyAny>) -> PyResult<Vec<Structure>> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(vec![structure(&s)?]);
    }
    let names: Vec<String> = obj
        .py()
        .import("numpy")?
        .call_method1("asarray", (obj, "object"))?
        .call_method1("reshape", (-1i64,))?
        .call_method0("tolist")?
        .extract()
        .map_err(|_| invalid("structures must be a name or an array of names"))?;
    names.iter().map(|s| structure(s)).collect()
}

fn complement(obj: Option<&Bound<PyAny>>, k: usize) -> PyResult<Vec<usize>> {
    match obj.filter(|o| !o.is_none()) {
        None => Ok((0..k).collect()),
        Some(o) => o
            .extract()
            .map_err(|_| invalid("complement must be one column index per row")),
    }
}

fn finite_lags(h: &[f64]) -> PyResult<()> {
    match h.iter().find(|v| !v.is_finite()) {
        Some(v) => Err(invalid(format!("lags must be finite, got {v}"))),
        None => Ok(()),
    }
}

/// `(n, k, k)` stack of `f(i)`, refused before allocating when it exceeds the available memory.
fn stack<'py>(
    py: Python<'py>,
    n: usize,
    k: usize,
    f: impl Fn(usize) -> Mat + Sync,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let bytes = (n as u64).saturating_mul((k * k) as u64).saturating_mul(8);
    memory::check(
        bytes,
        None,
        &format!("{n} transition matrices of {k} x {k}"),
    )
    .map_err(to_rust)?;
    let mut a = Array3::<f64>::zeros((n, k, k));
    py.detach(|| {
        for (i, mut s) in a.outer_iter_mut().enumerate() {
            let m = f(i);
            s.indexed_iter_mut().for_each(|((r, c), v)| *v = m[(r, c)]);
        }
    });
    Ok(a.into_pyarray(py))
}

/// One-directional transiogram model: a continuous-lag Markov chain (``markov``) or one
/// analytical curve per entry (``entrywise``, Li 2007).
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> exp = tg.downhole_transiogram(
/// ...     d["CODE"], lags=[0.5, 1.0, 1.5, 2.0], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> markov = tg.TransiogramModel.fit_markov(exp)
/// >>> markov.kind, markov.transition(1.0).shape, markov.transition([1.0, 2.0]).shape
/// ('markov', (3, 3), (2, 3, 3))
/// >>> bool(np.nanmax(np.abs(markov.residuals(exp))) < 0.05)
/// True
/// >>> chain = tg.MarkovChain().fit(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> tg.TransiogramModel.markov(chain).rates.shape
/// (3, 3)
/// >>> curves = tg.TransiogramModel.fit_entrywise(exp, markov.proportions, structures="spherical")
/// >>> curves.kind, curves.rates
/// ('entrywise', None)
/// >>> fixed = tg.TransiogramModel.entrywise([0.5, 0.5], "exponential", [[2.0, 2.0], [2.0, 2.0]], max_lag=10.0)
/// >>> np.round(fixed.transition(100.0), 3)
/// array([[0.5, 0.5],
///        [0.5, 0.5]])
/// >>> tg.TransiogramModel.from_json(markov.to_json()) == markov
/// True
#[pyclass(
    module = "transiograms",
    name = "TransiogramModel",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyTransiogramModel(pub Model1D);

#[pymethods]
impl PyTransiogramModel {
    /// Markov model ``T(h) = exp(R h)`` from a rate matrix or a chain with a generator.
    ///
    /// Parameters
    /// ----------
    /// rates : array_like or MarkovChain
    ///     ``(k, k)`` generator, or a chain that has one.
    ///
    /// Returns
    /// -------
    /// TransiogramModel
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``rates`` is not a generator matrix or the chain has none.
    #[staticmethod]
    fn markov(rates: &Bound<PyAny>) -> PyResult<Self> {
        let r = match rates.cast::<PyMarkovChain>() {
            Ok(c) => c.borrow().inner()?.generator.clone().ok_or_else(|| {
                invalid("this chain has no generator: fit it embedded or build it from rates")
            })?,
            Err(_) => RateMatrix::new(mat(rates, "rates")?).map_err(to_rust)?,
        };
        Model1D::markov(r).map(Self).map_err(to_rust)
    }

    /// Entrywise model: each entry rises from 0 (or its nugget) to its sill ``p_j`` with an
    /// exponential, spherical or Gaussian shape; the complement entry of each row closes it to 1.
    ///
    /// Parameters
    /// ----------
    /// proportions : array_like
    /// structures : str or array_like
    ///     One shape for all entries, or ``(k, k)`` names.
    /// ranges : array_like
    ///     ``(k, k)`` practical ranges.
    /// nuggets : array_like, optional
    ///     ``(k, k)`` nuggets in [0, 1).
    /// complement : array_like, optional
    ///     Complement column per row; the diagonal by default.
    /// max_lag : float
    ///     Validity (non-negative entries) is checked on a lag grid up to here.
    ///
    /// Returns
    /// -------
    /// TransiogramModel
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When a shape or name is invalid, a range is not positive, a nugget is outside [0, 1),
    ///     or the complement entries turn negative before ``max_lag``.
    #[staticmethod]
    #[pyo3(signature = (proportions, structures, ranges, *, nuggets=None, complement=None, max_lag))]
    fn entrywise(
        proportions: &Bound<PyAny>,
        structures: &Bound<PyAny>,
        ranges: &Bound<PyAny>,
        nuggets: Option<&Bound<PyAny>>,
        complement: Option<&Bound<PyAny>>,
        max_lag: f64,
    ) -> PyResult<Self> {
        let p = floats(proportions, "proportions")?;
        let k = p.len();
        let s = self::structures(structures)?;
        let r = mat(ranges, "ranges")?;
        let n = nuggets
            .filter(|o| !o.is_none())
            .map(|o| mat(o, "nuggets"))
            .transpose()?
            .unwrap_or_else(|| Mat::zeros(k, k));
        if r.shape() != (k, k) || n.shape() != (k, k) || !(s.len() == 1 || s.len() == k * k) {
            return Err(invalid(format!(
                "ranges and nuggets must be {k} x {k} and structures 1 or {k} x {k}"
            )));
        }
        let c = self::complement(complement, k)?;
        if c.len() != k {
            return Err(invalid(format!(
                "give {k} complement columns, not {}",
                c.len()
            )));
        }
        let entries = (0..k * k)
            .map(|e| {
                let (i, j) = (e / k, e % k);
                (c[i] != j).then(|| Entry {
                    structure: s[if s.len() == 1 { 0 } else { e }],
                    range: r[(i, j)],
                    nugget: n[(i, j)],
                })
            })
            .collect();
        Model1D::entrywise(p, entries, c, max_lag)
            .map(Self)
            .map_err(to_rust)
    }

    /// Markov model fitted to an experimental transiogram by weighted least squares.
    ///
    /// Parameters
    /// ----------
    /// experimental : ExperimentalTransiogram
    /// proportions : array_like, optional
    ///     Sills to honour. By default the sills are the stationary proportions of the starting
    ///     rates (``start``, or the rates read off the first lag where every category is a tail),
    ///     which can differ from the data proportions; pass the sample proportions to hold the
    ///     sills there.
    /// background : int
    ///     Category whose rates are filled from the others; -1 means the last.
    /// start : array_like, optional
    ///     Starting rates.
    ///
    /// Returns
    /// -------
    /// TransiogramModel
    ///     ``kind="markov"``; ``residuals(experimental)`` gives its misfit.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the proportions do not sum to 1, ``background`` is not a category, or no lag has
    ///     every category as a tail and ``start`` is not given.
    /// ConvergenceError
    ///     When no valid starting rates exist for these proportions, or the simplex search does
    ///     not converge within its evaluation budget.
    #[staticmethod]
    #[pyo3(
        signature = (experimental, *, proportions=None, background=-1, start=None),
        text_signature = "(experimental, *, proportions=None, background=-1, start=None)"
    )]
    fn fit_markov(
        py: Python,
        experimental: &PyExperimental,
        proportions: Option<&Bound<PyAny>>,
        background: i64,
        start: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let p = proportions
            .filter(|o| !o.is_none())
            .map(|o| floats(o, "proportions"))
            .transpose()?;
        let b = background_index(background, experimental.k())?;
        let s = start
            .filter(|o| !o.is_none())
            .map(|o| RateMatrix::new(mat(o, "start")?).map_err(to_rust))
            .transpose()?;
        let e = &experimental.0;
        py.detach(|| Model1D::fit_markov(e, p.as_deref(), b, s.as_ref()))
            .map(Self)
            .map_err(to_rust)
    }

    /// Entrywise model fitted entry by entry (range by golden section, nugget 0).
    ///
    /// Parameters
    /// ----------
    /// experimental : ExperimentalTransiogram
    /// proportions : array_like
    ///     Sills ``(k,)``.
    /// structures : str or sequence of str, optional
    ///     One shape for all entries or ``k * k`` names (row-major, complement entries ignored);
    ///     None means "exponential".
    /// complement : array_like, optional
    ///     Complement column per row; the diagonal by default.
    ///
    /// Returns
    /// -------
    /// TransiogramModel
    ///     ``kind="entrywise"``; ``residuals(experimental)`` gives its misfit.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the proportions do not sum to 1, the number of structures is not 1 or ``k * k``,
    ///     or the experimental transiogram has no usable row.
    #[staticmethod]
    #[pyo3(signature = (experimental, proportions, *, structures=None, complement=None))]
    fn fit_entrywise(
        py: Python,
        experimental: &PyExperimental,
        proportions: &Bound<PyAny>,
        structures: Option<&Bound<PyAny>>,
        complement: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let p = floats(proportions, "proportions")?;
        let s = match structures.filter(|o| !o.is_none()) {
            Some(o) => self::structures(o)?,
            None => vec![Structure::Exponential],
        };
        let c = self::complement(complement, p.len())?;
        let e = &experimental.0;
        py.detach(|| Model1D::fit_entrywise(e, &p, &s, &c))
            .map(Self)
            .map_err(to_rust)
    }

    /// "markov" or "entrywise".
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0 {
            Model1D::Markov { .. } => "markov",
            Model1D::Entrywise { .. } => "entrywise",
        }
    }
    /// ``(k,)`` proportions (the sills).
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.proportions().to_vec())
    }
    /// Rate matrix of a Markov model, else None.
    #[getter]
    fn rates<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match &self.0 {
            Model1D::Markov { rates, .. } => Some(mat2(py, rates.matrix())),
            Model1D::Entrywise { .. } => None,
        }
    }
    /// Transition probabilities ``T(h)``; negative lags by reversal.
    ///
    /// Parameters
    /// ----------
    /// h : float or array_like
    ///     Lag or lags.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k, k)`` for a scalar, ``(n, k, k)`` for an array.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When a lag is not finite, or before allocating a stack larger than the available memory.
    fn transition<'py>(
        &self,
        py: Python<'py>,
        h: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (hs, scalar) = scalar_or_floats(h, "h")?;
        finite_lags(&hs)?;
        if scalar {
            return Ok(mat2(py, &self.0.transition(hs[0])).into_any());
        }
        let k = self.0.proportions().len();
        Ok(stack(py, hs.len(), k, |i| self.0.transition(hs[i]))?.into_any())
    }
    /// Fit-quality report against an experimental transiogram: experimental minus model ``t_ij``
    /// at each of its lags.
    ///
    /// The model is evaluated at the mean lag of each lag class's pairs (the nominal lag when a
    /// class is empty), as ``fit_markov`` and ``fit_entrywise`` do; those fits minimise the sum of
    /// these squared residuals weighted by the tail totals.
    ///
    /// Parameters
    /// ----------
    /// experimental : ExperimentalTransiogram
    ///     With as many categories as the model.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(n_lags, k, k)`` residuals; NaN where the experimental row has no pairs.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the category counts differ.
    ///
    /// Examples
    /// --------
    /// >>> d = tg.datasets.varved_sequence()
    /// >>> exp = tg.downhole_transiogram(
    /// ...     d["CODE"], lags=[0.5, 1.0, 2.0], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
    /// >>> r = tg.TransiogramModel.fit_markov(exp).residuals(exp)
    /// >>> r.shape
    /// (3, 3, 3)
    /// >>> round(float(np.nanmax(np.abs(r))), 2)
    /// 0.05
    fn residuals<'py>(
        &self,
        py: Python<'py>,
        experimental: PyRef<'py, PyExperimental>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let r = self.0.residuals(&experimental.0).map_err(to_rust)?;
        let k = self.0.proportions().len();
        stack(py, r.len(), k, |i| r[i].clone())
    }
    /// JSON text of this object; ``TransiogramModel.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// TransiogramModel from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// TransiogramModel
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not TransiogramModel JSON or describes an invalid TransiogramModel.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        unjson(text, "TransiogramModel").map(Self)
    }
    fn __repr__(&self) -> String {
        format!(
            "TransiogramModel(kind={:?}, n_categories={})",
            self.kind(),
            self.0.proportions().len()
        )
    }
}

fn rate_matrix(obj: &Bound<PyAny>, what: &str) -> PyResult<RateMatrix> {
    RateMatrix::new(mat(obj, what)?).map_err(to_rust)
}

/// 3D Markov transition model (Carle and Fogg): proportions and one rate matrix per principal
/// axis, interpolated in between, rotated by azimuth, dip and rake.
///
/// Parameters
/// ----------
/// proportions : array_like
/// rates_x, rates_y, rates_z : array_like
///     ``(k, k)`` generators along the principal +x', +y' and +z' (up) axes, each with
///     ``p^T R = 0``.
/// rotation : tuple of 3 float
///     Azimuth, dip and rake in degrees.
/// background : int
///     Background category; -1 means the last.
///
/// Raises
/// ------
/// InvalidInput
///     When the proportions do not sum to 1, a rate matrix is not a generator with
///     ``p^T R = 0``, or ``background`` is not a category.
///
/// Examples
/// --------
/// >>> m = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> m.n_categories, m.background
/// (3, 2)
/// >>> np.round(m.mean_lengths([0.0, 0.0, 1.0]), 6)
/// array([4., 3., 2.])
/// >>> m.transition([0.0, 0.0, 1.0]).shape, m.transition([[0.0, 0.0, 1.0], [10.0, 0.0, 0.0]]).shape
/// ((3, 3), (2, 3, 3))
/// >>> m.rates([1.0, 0.0, 0.0]).shape
/// (3, 3)
/// >>> c = m.closeness([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
/// >>> float(c[0]), bool(c[1] < 1.0)
/// (1.0, True)
/// >>> turned = tg.TransitionModel(m.proportions, m.rates_x, m.rates_y, m.rates_z, rotation=(30.0, 0.0, 0.0))
/// >>> turned.rotation
/// (30.0, 0.0, 0.0)
/// >>> tg.TransitionModel.from_json(m.to_json()) == m
/// True
#[pyclass(
    module = "transiograms",
    name = "TransitionModel",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyTransitionModel(pub TransitionModel);

#[pymethods]
impl PyTransitionModel {
    #[new]
    #[pyo3(
        signature = (proportions, rates_x, rates_y, rates_z, *, rotation=(0.0, 0.0, 0.0), background=-1),
        text_signature = "(proportions, rates_x, rates_y, rates_z, *, rotation=(0.0, 0.0, 0.0), background=-1)"
    )]
    fn new(
        proportions: &Bound<PyAny>,
        rates_x: &Bound<PyAny>,
        rates_y: &Bound<PyAny>,
        rates_z: &Bound<PyAny>,
        rotation: (f64, f64, f64),
        background: i64,
    ) -> PyResult<Self> {
        let p = floats(proportions, "proportions")?;
        let b = background_index(background, p.len())?;
        TransitionModel::new(
            p,
            [
                rate_matrix(rates_x, "rates_x")?,
                rate_matrix(rates_y, "rates_y")?,
                rate_matrix(rates_z, "rates_z")?,
            ],
            [rotation.0, rotation.1, rotation.2],
            b,
        )
        .map(Self)
        .map_err(to_rust)
    }

    /// Model from proportions, mean lengths along each principal axis and optional embedded
    /// juxtapositions per axis (maximum entropy without them).
    ///
    /// Parameters
    /// ----------
    /// proportions : array_like
    /// mean_lengths : array_like
    ///     ``(3, k)``: rows x', y', z'.
    /// embedded_x, embedded_y, embedded_z : array_like, optional
    ///     ``(k, k)`` embedded juxtapositions along each axis.
    /// rotation : tuple of 3 float
    /// background : int
    ///
    /// Returns
    /// -------
    /// TransitionModel
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the proportions do not sum to 1, a mean length is not positive, or the
    ///     juxtapositions give no valid generator.
    #[staticmethod]
    #[pyo3(
        signature = (proportions, mean_lengths, *, embedded_x=None, embedded_y=None, embedded_z=None, rotation=(0.0, 0.0, 0.0), background=-1),
        text_signature = "(proportions, mean_lengths, *, embedded_x=None, embedded_y=None, embedded_z=None, rotation=(0.0, 0.0, 0.0), background=-1)"
    )]
    fn from_geology(
        proportions: &Bound<PyAny>,
        mean_lengths: &Bound<PyAny>,
        embedded_x: Option<&Bound<PyAny>>,
        embedded_y: Option<&Bound<PyAny>>,
        embedded_z: Option<&Bound<PyAny>>,
        rotation: (f64, f64, f64),
        background: i64,
    ) -> PyResult<Self> {
        let p = floats(proportions, "proportions")?;
        let k = p.len();
        let l = mat(mean_lengths, "mean_lengths")?;
        if l.shape() != (3, k) {
            return Err(invalid(format!(
                "mean_lengths must be (3, {k}), not {:?}",
                l.shape()
            )));
        }
        let rows: Vec<Vec<f64>> = (0..3).map(|a| l.row(a).iter().copied().collect()).collect();
        let e = [embedded_x, embedded_y, embedded_z].map(|o| {
            o.filter(|o| !o.is_none())
                .map(|o| mat(o, "embedded"))
                .transpose()
        });
        let [ex, ey, ez] = e;
        let (ex, ey, ez) = (ex?, ey?, ez?);
        TransitionModel::from_geology(
            &p,
            [&rows[0], &rows[1], &rows[2]],
            [ex.as_ref(), ey.as_ref(), ez.as_ref()],
            background_index(background, k)?,
            [rotation.0, rotation.1, rotation.2],
        )
        .map(Self)
        .map_err(to_rust)
    }

    /// ``(k,)`` proportions.
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.proportions.clone())
    }
    /// ``(k, k)`` rate matrix along the rotated x axis.
    #[getter]
    fn rates_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat2(py, self.0.rates[0].matrix())
    }
    /// ``(k, k)`` rate matrix along the rotated y axis.
    #[getter]
    fn rates_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat2(py, self.0.rates[1].matrix())
    }
    /// ``(k, k)`` rate matrix along the rotated z axis.
    #[getter]
    fn rates_z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat2(py, self.0.rates[2].matrix())
    }
    /// ``(azimuth, dip, rake)`` in degrees.
    #[getter]
    fn rotation(&self) -> (f64, f64, f64) {
        let r = self.0.rotation;
        (r[0], r[1], r[2])
    }
    /// Background category whose rates are filled from the others.
    #[getter]
    fn background(&self) -> usize {
        self.0.background
    }
    /// Number of categories ``k``.
    #[getter]
    fn n_categories(&self) -> usize {
        self.0.k()
    }

    /// Transition probabilities ``T(h)`` at lag vectors.
    ///
    /// Parameters
    /// ----------
    /// lags : array_like
    ///     ``(3,)`` or ``(n, 3)``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k, k)`` for one lag, ``(n, k, k)`` for several.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``lags`` has another shape, or before allocating a stack larger than the available
    ///     memory.
    fn transition<'py>(
        &self,
        py: Python<'py>,
        lags: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (h, single) = lags3(lags, "lags")?;
        finite_lags(&h.concat())?;
        if single {
            return Ok(mat2(py, &self.0.transition(h[0])).into_any());
        }
        Ok(stack(py, h.len(), self.0.k(), |i| self.0.transition(h[i]))?.into_any())
    }

    /// Generator interpolated along a non-zero lag vector.
    ///
    /// Parameters
    /// ----------
    /// lag : sequence of 3 float
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k, k)``.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``lag`` is zero or not finite.
    fn rates<'py>(&self, py: Python<'py>, lag: [f64; 3]) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat2(py, self.0.rates_along(lag).map_err(to_rust)?.matrix()))
    }

    /// Closeness ``det T(h)^(1 / (k - 1))`` per lag: 1 at lag 0, falling with distance.
    ///
    /// Parameters
    /// ----------
    /// lags : array_like
    ///     ``(3,)`` or ``(n, 3)``.
    ///
    /// Returns
    /// -------
    /// float or numpy.ndarray
    ///     One value per lag.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``lags`` has another shape.
    fn closeness<'py>(
        &self,
        py: Python<'py>,
        lags: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (h, single) = lags3(lags, "lags")?;
        finite_lags(&h.concat())?;
        let c: Vec<f64> = py.detach(|| h.iter().map(|&h| self.0.closeness(h)).collect());
        if single {
            return Ok(pyo3::types::PyFloat::new(py, c[0]).into_any());
        }
        Ok(vec1(py, c).into_any())
    }

    /// Mean lengths of the categories along a lag direction.
    ///
    /// Parameters
    /// ----------
    /// lag : array_like
    ///     Non-zero direction vector ``(3,)``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k,)``.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``lag`` is zero or not ``(3,)``.
    fn mean_lengths<'py>(
        &self,
        py: Python<'py>,
        lag: [f64; 3],
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        finite_lags(&lag)?;
        Ok(vec1(py, self.0.mean_lengths(lag)))
    }

    /// JSON text of this object; ``TransitionModel.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// TransitionModel from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// TransitionModel
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not TransitionModel JSON or describes an invalid TransitionModel.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let m: TransitionModel = unjson(text, "TransitionModel")?;
        TransitionModel::new(m.proportions, m.rates, m.rotation, m.background)
            .map(Self)
            .map_err(to_rust)
    }
    fn __repr__(&self) -> String {
        format!(
            "TransitionModel(n_categories={}, rotation={:?}, background={})",
            self.0.k(),
            self.0.rotation,
            self.0.background
        )
    }
}

/// How far an ensemble of realizations sits from its model's proportions and transiograms.
///
/// Attributes
/// ----------
/// target_proportions : numpy.ndarray
///     ``(k,)`` model proportions.
/// proportions : numpy.ndarray
///     ``(n, k)`` proportions of each realization over its active cells.
/// mean_proportions : numpy.ndarray
///     ``(k,)`` ensemble proportions.
/// offsets : numpy.ndarray
///     ``(m, 3)`` cell offsets ``l * direction`` for ``l = 1..n_lags``, direction-major.
/// model, mean, standard_error : numpy.ndarray
///     ``(m, k, k)`` model ``t_ij``, ensemble mean ``t_ij`` and its standard error (spread
///     across realizations over the square root of their number); NaN where a tail category
///     has no pair, and standard errors are NaN with one realization.
/// proportion_error : float
///     Largest ``|mean proportion - target|``.
/// transiogram_error : float
///     Largest ``|mean t_ij - model t_ij|`` over every entry and offset.
/// max_standard_errors : float
///     Largest ``|mean t_ij - model t_ij| / se``, with standard errors below 0.001 counted as
///     0.001; NaN with one realization. Values above about 4 mean the ensemble misses the
///     model by more than sampling error.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> real = tg.mcrf_simulate(model, grid, np.empty((0, 3)), [], n=4, seed=1)
/// >>> q = tg.fit_quality(model, grid, real, n_lags=2)
/// >>> q.offsets.shape, q.model.shape == q.mean.shape == q.standard_error.shape
/// ((8, 3), True)
/// >>> q.proportions.shape, q.mean_proportions.shape, q.target_proportions
/// ((4, 3), (3,), array([0.5, 0.3, 0.2]))
/// >>> bool(q.proportion_error < 0.2 and q.transiogram_error < 0.5 and q.max_standard_errors > 0)
/// True
/// >>> tg.FitQuality.from_json(q.to_json()).to_json() == q.to_json()
/// True
#[pyclass(module = "transiograms", name = "FitQuality", frozen)]
pub struct PyFitQuality(pub FitQuality);

#[pymethods]
impl PyFitQuality {
    /// ``(k,)`` model proportions.
    #[getter]
    fn target_proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.target_proportions.clone())
    }
    /// ``(n, k)`` proportions of each realization over its active cells.
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = (self.0.n_realizations, self.0.k);
        Array2::from_shape_vec(shape, self.0.proportions.clone())
            .map(|a| a.into_pyarray(py))
            .map_err(invalid)
    }
    /// ``(k,)`` ensemble proportions.
    #[getter]
    fn mean_proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.mean_proportions())
    }
    /// ``(m, 3)`` cell offsets ``l * direction`` for ``l = 1..n_lags``, direction-major.
    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let o = &self.0.offsets;
        Array2::from_shape_fn((o.len(), 3), |(i, a)| o[i][a]).into_pyarray(py)
    }
    /// ``(m, k, k)`` model ``t_ij`` at each offset.
    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        mat3(py, &self.0.model, self.0.k, self.0.k)
    }
    /// ``(m, k, k)`` ensemble mean ``t_ij``; NaN where a tail category has no pair.
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        mat3(py, &self.0.mean, self.0.k, self.0.k)
    }
    /// ``(m, k, k)`` standard error of ``mean`` (spread over the square root of the number of realizations); NaN with one realization.
    #[getter]
    fn standard_error<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        mat3(py, &self.0.standard_error, self.0.k, self.0.k)
    }
    /// Largest ``|mean proportion - target|``.
    #[getter]
    fn proportion_error(&self) -> f64 {
        self.0.proportion_error()
    }
    /// Largest ``|mean t_ij - model t_ij|`` over every entry and offset.
    #[getter]
    fn transiogram_error(&self) -> f64 {
        self.0.transiogram_error()
    }
    /// Largest ``|mean t_ij - model t_ij| / se`` (se below 0.001 counted as 0.001); above about 4 the ensemble misses the model by more than sampling error.
    #[getter]
    fn max_standard_errors(&self) -> f64 {
        self.0.max_standard_errors()
    }
    /// JSON text of this object; ``FitQuality.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// FitQuality from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// FitQuality
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not FitQuality JSON or describes an invalid FitQuality.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let q: FitQuality = unjson(text, "FitQuality")?;
        let (k, m) = (q.k, q.offsets.len());
        let square = |ms: &[transiograms_core::Mat]| {
            ms.len() == m && ms.iter().all(|t| t.nrows() == k && t.ncols() == k)
        };
        if k == 0
            || q.target_proportions.len() != k
            || q.n_realizations.checked_mul(k) != Some(q.proportions.len())
            || !square(&q.model)
            || !square(&q.mean)
            || !square(&q.standard_error)
        {
            return Err(invalid(
                "FitQuality JSON needs k >= 1, k target proportions, n * k proportions and one k x k model, mean and standard_error per offset",
            ));
        }
        Ok(Self(q))
    }
    fn __repr__(&self) -> String {
        format!(
            "FitQuality(n={}, k={}, offsets={}, proportion_error={:.4}, transiogram_error={:.4}, max_standard_errors={:.1})",
            self.0.n_realizations,
            self.0.k,
            self.0.offsets.len(),
            self.0.proportion_error(),
            self.0.transiogram_error(),
            self.0.max_standard_errors()
        )
    }
}

/// Fit-quality report: ensemble proportions and grid transiograms of ``realizations`` against
/// ``model``, per realization and on average, with standard errors from the spread across
/// realizations.
///
/// Parameters
/// ----------
/// model : TransitionModel
/// grid : Grid
/// realizations : Realizations
///     From ``tprogs_simulate`` or ``mcrf_simulate`` on ``grid``.
/// directions : sequence of (int, int, int)
///     Cell steps of lag 1; +x, +y, +z and -z by default.
/// n_lags : int
///     Lags ``1..n_lags`` along each direction.
/// mask : array_like, optional
///     Active cells.
///
/// Returns
/// -------
/// FitQuality
///
/// Raises
/// ------
/// InvalidInput
///     When ``realizations`` has another number of cells or categories than ``grid`` and
///     ``model``, ``n_lags`` is 0, a direction is zero, or ``mask`` has another length.
///
/// Examples
/// --------
/// >>> model = tg.TransitionModel.from_geology(
/// ...     [0.5, 0.3, 0.2], [[40.0, 30.0, 20.0], [40.0, 30.0, 20.0], [4.0, 3.0, 2.0]])
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (12, 12, 6))
/// >>> xyz, codes = [[55.0, 55.0, 0.5], [55.0, 55.0, 3.5], [5.0, 105.0, 2.5]], [0, 1, 2]
/// >>> cells, hard = grid.snap(xyz, codes)
/// >>> real = tg.mcrf_simulate(model, grid, xyz, codes, n=2, seed=1)
/// >>> tg.fit_quality(model, grid, real, directions=[(0, 0, 1)], n_lags=3).offsets
/// array([[0, 0, 1],
///        [0, 0, 2],
///        [0, 0, 3]])
#[pyfunction]
#[pyo3(name = "fit_quality", signature = (model, grid, realizations, *, directions=None, n_lags=6, mask=None))]
fn fit_quality_report<'py>(
    py: Python<'py>,
    model: &PyTransitionModel,
    grid: &PyGrid,
    realizations: &PyRealizations,
    directions: Option<Vec<[i64; 3]>>,
    n_lags: usize,
    mask: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyFitQuality> {
    let dirs = directions.unwrap_or_else(|| vec![[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]);
    let m = self::mask(mask, grid.0.n_cells())?;
    py.detach(|| {
        fit_quality(
            &model.0,
            &grid.0,
            m.as_deref(),
            &realizations.r,
            &dirs,
            n_lags,
        )
    })
    .map(PyFitQuality)
    .map_err(to_rust)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyFitQuality>()?;
    m.add_function(wrap_pyfunction!(fit_quality_report, m)?)?;
    m.add_class::<PyExperimental>()?;
    m.add_class::<PyTransiogramModel>()?;
    m.add_class::<PyTransitionModel>()?;
    m.add_function(wrap_pyfunction!(experimental_transiogram, m)?)?;
    m.add_function(wrap_pyfunction!(downhole_transiogram, m)?)?;
    m.add_function(wrap_pyfunction!(grid_transiogram, m)?)?;
    Ok(())
}
