//! `MarkovChain`, `TestResult`, `Entropy`, `Sojourn` and the order, homogeneity and sojourn
//! functions.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use transiograms_chain::{
    FitOptions, Hard, Kind, MarkovChain, Method, Simulated, Start, Target, TestResult,
};
use transiograms_core::{Code, Direction, EmptyRow, RateMatrix, SequenceSet, TransitionMatrix};

use crate::args::{
    SequenceArgs, codes, direction, floats, hole_ids, mat, mat2, scalar_or_floats,
    sequence_set_step, to_rust, vec1,
};
use crate::core::{json, same_json, unjson};
use crate::invalid;

fn method(s: &str) -> PyResult<Method> {
    match s {
        "g" => Ok(Method::G),
        "pearson" => Ok(Method::Pearson),
        _ => Err(invalid(format!(
            "method must be \"g\" or \"pearson\", not {s:?}"
        ))),
    }
}

fn direction_name(d: Direction) -> &'static str {
    match d {
        Direction::Up => "up",
        Direction::Down => "down",
    }
}

fn kind_name(k: &str) -> PyResult<&str> {
    match k {
        "embedded" | "fixed" => Ok(k),
        _ => Err(invalid(format!(
            "kind must be \"embedded\" or \"fixed\", not {k:?}"
        ))),
    }
}

/// `kind` with its step: the chain's own `step`, else the sampling step of the data.
fn kind_of(kind: &str, step: Option<f64>, sampled: Option<f64>) -> PyResult<Kind> {
    match kind {
        "embedded" => Ok(Kind::Embedded),
        _ => step
            .or(sampled)
            .map(|step| Kind::Fixed { step })
            .ok_or_else(|| invalid("a fixed chain on intervals needs step=")),
    }
}

/// Result of a statistical test.
///
/// Attributes
/// ----------
/// name : str
///     "independence", "order 2 vs 1", "symmetry", "homogeneity" or "exponential".
/// method : str
///     "g", "pearson", "bowker" or "ks".
/// statistic : float
/// df : int
///     Degrees of freedom; 0 for "ks".
/// p_value : float
/// expected : numpy.ndarray or None
///     Expected counts under the null hypothesis when the test is tabular.
/// small_expected : float
///     Fraction of allowed cells whose expected count is below 5.
///
/// Examples
/// --------
/// >>> codes = tg.MarkovChain.from_matrix([[0.9, 0.1], [0.2, 0.8]]).simulate(500, seed=1)[0]
/// >>> r = tg.MarkovChain(kind="fixed", step=1.0).fit(codes).independence_test()
/// >>> r.name, r.method, r.df
/// ('independence', 'g', 1)
/// >>> bool(r.p_value < 0.001)
/// True
/// >>> tg.TestResult.from_json(r.to_json()).to_json() == r.to_json()
/// True
#[pyclass(
    module = "transiograms",
    name = "TestResult",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyTestResult(pub TestResult);

#[pymethods]
impl PyTestResult {
    /// "independence", "order 2 vs 1", "symmetry", "homogeneity" or "exponential".
    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }
    /// "g", "pearson", "bowker" or "ks".
    #[getter]
    fn method(&self) -> &str {
        &self.0.method
    }
    /// Test statistic.
    #[getter]
    fn statistic(&self) -> f64 {
        self.0.statistic
    }
    /// Degrees of freedom; 0 for "ks".
    #[getter]
    fn df(&self) -> usize {
        self.0.df
    }
    /// p-value of ``statistic`` under the null hypothesis.
    #[getter]
    fn p_value(&self) -> f64 {
        self.0.p_value
    }
    /// Expected counts under the null hypothesis for a tabular test, else None.
    #[getter]
    fn expected<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0.expected.as_ref().map(|m| mat2(py, m))
    }
    /// Fraction of allowed cells whose expected count is below 5.
    #[getter]
    fn small_expected(&self) -> f64 {
        self.0.small_expected
    }
    /// JSON text of this object; ``TestResult.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// TestResult from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// TestResult
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not TestResult JSON or describes an invalid TestResult.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        unjson(text, "TestResult").map(Self)
    }
    fn __repr__(&self) -> String {
        format!(
            "TestResult(name={:?}, method={:?}, statistic={:.6}, df={}, p_value={:.4e})",
            self.0.name, self.0.method, self.0.statistic, self.0.df, self.0.p_value
        )
    }
}

/// Entropies of a chain, in bits.
///
/// Attributes
/// ----------
/// post, pre : numpy.ndarray
///     Entropy of the row (what follows each category) and of the column (what precedes it).
/// post_normalized, pre_normalized : numpy.ndarray
///     The same divided by their maximum.
/// chain : float
///     Entropy rate of the whole chain.
/// max : float
///     Its maximum for this number of categories.
///
/// Examples
/// --------
/// >>> e = tg.MarkovChain.from_matrix([[0.0, 1.0], [1.0, 0.0]], kind="embedded").entropy()
/// >>> e.post, e.chain
/// (array([0., 0.]), 0.0)
/// >>> tg.Entropy.from_json(e.to_json()) == e
/// True
#[pyclass(module = "transiograms", name = "Entropy", frozen)]
pub struct PyEntropy(transiograms_chain::Entropy);

#[pymethods]
impl PyEntropy {
    /// ``(k,)`` entropy in bits of what follows each category (row entropy).
    #[getter]
    fn post<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.post.clone())
    }
    /// ``(k,)`` entropy in bits of what precedes each category (column entropy).
    #[getter]
    fn pre<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.pre.clone())
    }
    /// ``post`` divided by its maximum, in [0, 1].
    #[getter]
    fn post_normalized<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.post_normalized.clone())
    }
    /// ``pre`` divided by its maximum, in [0, 1].
    #[getter]
    fn pre_normalized<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.pre_normalized.clone())
    }
    /// Entropy rate of the whole chain, in bits.
    #[getter]
    fn chain(&self) -> f64 {
        self.0.chain
    }
    /// Largest entropy rate for this number of categories, in bits.
    #[getter]
    fn max(&self) -> f64 {
        self.0.max
    }
    /// JSON text of this object; ``Entropy.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// Entropy from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Entropy
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not Entropy JSON or describes an invalid Entropy.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let e: transiograms_chain::Entropy = unjson(text, "Entropy")?;
        let m = e.post.len();
        if [&e.pre, &e.post_normalized, &e.pre_normalized]
            .iter()
            .any(|v| v.len() != m)
        {
            return Err(invalid(format!(
                "Entropy JSON needs {m} values in pre, post_normalized and pre_normalized, as in post"
            )));
        }
        Ok(Self(e))
    }
    fn __eq__(&self, other: PyRef<Self>) -> PyResult<bool> {
        same_json(&self.0, &other.0)
    }
    fn __repr__(&self) -> String {
        format!("Entropy(chain={}, max={})", self.0.chain, self.0.max)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Settings {
    kind: String,
    direction: Direction,
    step: Option<f64>,
    self_transitions: bool,
    pseudocount: f64,
    on_empty_row: EmptyRow,
}

/// Markov chain of categories along drillholes.
///
/// ``kind="embedded"`` counts transitions between successive beds (diagonal zero) and estimates
/// mean bed lengths and the continuous-lag generator ``Q`` from bed thicknesses.
/// ``kind="fixed"`` counts transitions between samples one ``step`` apart.
///
/// Parameters
/// ----------
/// kind : {"embedded", "fixed"}
/// direction : {"up", "down"}
///     "up" runs from the deepest row to the shallowest (older to younger).
/// step : float, optional
///     Step of a fixed chain; defaults to the sampling step of the data.
/// self_transitions : bool
///     Embedded only: keep adjacent intervals of one category as separate beds.
/// pseudocount : float
///     Added to every allowed count.
/// on_empty_row : {"raise", "uniform"}
///     What a category with no transition out gives when ``pseudocount`` is 0.
///
/// Raises
/// ------
/// InvalidInput
///     When an option is unknown or ``pseudocount`` is negative.
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> chain = tg.MarkovChain().fit(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> chain.kind, chain.n_categories
/// ('embedded', 3)
/// >>> bool(np.allclose(chain.transition_matrix.sum(axis=1), 1.0))
/// True
/// >>> chain.transition_at(2.0).shape, chain.transition_at([1.0, 2.0]).shape
/// ((3, 3), (2, 3, 3))
/// >>> bool(chain.independence_test().p_value < 0.01)
/// True
/// >>> chain.symmetry_test().name
/// 'symmetry'
/// >>> e = chain.entropy()
/// >>> bool(0.0 < e.chain <= e.max)
/// True
/// >>> chain.difference_matrix().shape, chain.asymmetry().shape, chain.proportion_std(100).shape
/// ((3, 3), (3, 3), (3,))
/// >>> sorted(chain.simulate(10.0, n=2, seed=1))
/// ['code', 'from', 'realization', 'to']
/// >>> chain.simulate_beds(5, n=2, seed=1).shape
/// (2, 5)
/// >>> fixed = tg.MarkovChain.from_matrix([[0.9, 0.1], [0.2, 0.8]])
/// >>> fixed.simulate(5, seed=1).shape
/// (1, 5)
/// >>> np.round(tg.MarkovChain.from_rates([[-0.5, 0.5], [0.25, -0.25]]).proportions, 4)
/// array([0.3333, 0.6667])
/// >>> tg.MarkovChain.from_geology([0.5, 0.3, 0.2], [2.0, 1.0, 0.5]).kind
/// 'continuous'
/// >>> tg.MarkovChain.from_json(chain.to_json()) == chain
/// True
#[pyclass(module = "transiograms", name = "MarkovChain", eq, from_py_object)]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PyMarkovChain {
    settings: Settings,
    chain: Option<MarkovChain>,
}

impl PyMarkovChain {
    fn fitted(&self) -> PyResult<&MarkovChain> {
        self.chain
            .as_ref()
            .ok_or_else(|| invalid("the chain is not fitted; call fit first"))
    }

    fn built(chain: MarkovChain, kind: &str, step: Option<f64>) -> Self {
        Self {
            settings: Settings {
                kind: kind.to_string(),
                direction: chain.direction,
                step,
                self_transitions: false,
                pseudocount: 0.0,
                on_empty_row: EmptyRow::Raise,
            },
            chain: Some(chain),
        }
    }

    pub fn inner(&self) -> PyResult<&MarkovChain> {
        self.fitted()
    }

    fn hard<'py>(
        hard_from: Option<&Bound<'py, PyAny>>,
        hard_to: Option<&Bound<'py, PyAny>>,
        hard_codes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Hard> {
        let given = |o: Option<&Bound<'py, PyAny>>| o.filter(|o| !o.is_none()).cloned();
        match (given(hard_from), given(hard_to), given(hard_codes)) {
            (None, None, None) => Ok(Hard::default()),
            (Some(f), t, Some(c)) => {
                let from = floats(&f, "hard_from")?;
                let to = match t {
                    Some(t) => floats(&t, "hard_to")?,
                    None => from.clone(),
                };
                let codes = codes(&c, "hard_codes")?;
                if codes.contains(&transiograms_core::MISSING) {
                    return Err(invalid("hard_codes must not be missing"));
                }
                Ok(Hard { from, to, codes })
            }
            _ => Err(invalid(
                "give hard_from and hard_codes together (hard_to defaults to hard_from)",
            )),
        }
    }

    fn start(start: &Bound<PyAny>) -> PyResult<Start> {
        if let Ok(s) = start.extract::<String>() {
            return match s.as_str() {
                "stationary" => Ok(Start::Stationary),
                _ => Err(invalid(format!(
                    "start must be a code or \"stationary\", not {s:?}"
                ))),
            };
        }
        start
            .extract::<u32>()
            .map(Start::Code)
            .map_err(|_| invalid("start must be a code or \"stationary\""))
    }
}

fn codes2<'py>(py: Python<'py>, rows: &[Vec<Code>]) -> PyResult<Bound<'py, PyAny>> {
    let n = rows.len();
    let m = rows.first().map_or(0, Vec::len);
    let flat: Vec<i64> = rows.iter().flatten().map(|&c| c as i64).collect();
    Ok(PyArray1::from_vec(py, flat).reshape([n, m])?.into_any())
}

#[pymethods]
impl PyMarkovChain {
    #[new]
    #[pyo3(signature = (kind="embedded", *, direction="up", step=None, self_transitions=false, pseudocount=0.0, on_empty_row="raise"))]
    fn new(
        kind: &str,
        direction: &str,
        step: Option<f64>,
        self_transitions: bool,
        pseudocount: f64,
        on_empty_row: &str,
    ) -> PyResult<Self> {
        let on_empty_row = match on_empty_row {
            "raise" => EmptyRow::Raise,
            "uniform" => EmptyRow::Uniform,
            o => {
                return Err(invalid(format!(
                    "on_empty_row must be \"raise\" or \"uniform\", not {o:?}"
                )));
            }
        };
        Ok(Self {
            settings: Settings {
                kind: kind_name(kind)?.to_string(),
                direction: self::direction(direction)?,
                step,
                self_transitions,
                pseudocount,
                on_empty_row,
            },
            chain: None,
        })
    }

    /// Fits the chain to categorical sequences.
    ///
    /// Parameters
    /// ----------
    /// codes : array_like or str
    ///     Category code per row, ``0..k-1``, -1 or NaN for missing.
    /// holes, from_, to, depth, step, gap, categories, data
    ///     The shared sequence arguments: per-row hole ids; interval tops and bottoms; or regular
    ///     sample depths with their ``step``; ``gap`` is the largest gap that keeps a run;
    ///     ``categories`` gives ``k``; strings name columns of ``data``.
    ///
    /// Returns
    /// -------
    /// MarkovChain
    ///     This chain, fitted.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When a code is out of range, the sequence arguments disagree in length, a named
    ///     column is missing from ``data``, or a category has no transition out and
    ///     ``on_empty_row="raise"``.
    #[pyo3(signature = (codes, *, holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        codes: Bound<'py, PyAny>,
        holes: Option<Bound<'py, PyAny>>,
        from_: Option<Bound<'py, PyAny>>,
        to: Option<Bound<'py, PyAny>>,
        depth: Option<Bound<'py, PyAny>>,
        step: Option<f64>,
        gap: f64,
        categories: Option<Bound<'py, PyAny>>,
        data: Option<Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (seqs, sampled) = sequence_set_step(SequenceArgs {
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
        let s = &slf.settings;
        let opts = FitOptions {
            kind: kind_of(&s.kind, s.step, sampled)?,
            direction: s.direction,
            gap,
            self_transitions: s.self_transitions,
            pseudocount: s.pseudocount,
            on_empty_row: s.on_empty_row,
        };
        let chain = py
            .detach(|| MarkovChain::fit(&seqs, &opts))
            .map_err(to_rust)?;
        if let Kind::Fixed { step } = chain.kind {
            slf.settings.step = Some(step);
        }
        slf.chain = Some(chain);
        Ok(slf)
    }

    /// Chain from a transition matrix.
    ///
    /// Parameters
    /// ----------
    /// matrix : array_like
    ///     ``(k, k)`` row-stochastic matrix; zero diagonal for ``kind="embedded"``.
    /// kind : {"fixed", "embedded"}
    /// step : float
    ///     Step of a fixed chain.
    /// mean_lengths : array_like, optional
    ///     Embedded only: mean bed length per category, which gives the generator.
    /// direction : {"up", "down"}
    ///
    /// Returns
    /// -------
    /// MarkovChain
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``matrix`` is not row-stochastic, an embedded diagonal is not zero, or
    ///     ``mean_lengths`` has another length or a non-positive entry.
    #[staticmethod]
    #[pyo3(signature = (matrix, *, kind="fixed", step=1.0, mean_lengths=None, direction="up"))]
    fn from_matrix(
        matrix: &Bound<PyAny>,
        kind: &str,
        step: f64,
        mean_lengths: Option<&Bound<PyAny>>,
        direction: &str,
    ) -> PyResult<Self> {
        let kind = kind_name(kind)?;
        let p = TransitionMatrix::new(mat(matrix, "matrix")?).map_err(to_rust)?;
        let l = mean_lengths
            .filter(|o| !o.is_none())
            .map(|o| floats(o, "mean_lengths"))
            .transpose()?;
        let k = kind_of(kind, Some(step), None)?;
        let chain =
            MarkovChain::from_transition(p, k, l, self::direction(direction)?).map_err(to_rust)?;
        Ok(Self::built(chain, kind, (kind == "fixed").then_some(step)))
    }

    /// Continuous-lag chain from a generator (rate) matrix ``Q``.
    ///
    /// Parameters
    /// ----------
    /// rates : array_like
    ///     ``(k, k)`` with off-diagonals >= 0 and rows summing to 0.
    /// direction : {"up", "down"}
    ///
    /// Returns
    /// -------
    /// MarkovChain
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``rates`` is not a generator matrix.
    #[staticmethod]
    #[pyo3(signature = (rates, *, direction="up"))]
    fn from_rates(rates: &Bound<PyAny>, direction: &str) -> PyResult<Self> {
        let q = RateMatrix::new(mat(rates, "rates")?).map_err(to_rust)?;
        let chain = MarkovChain::from_rates(q, self::direction(direction)?).map_err(to_rust)?;
        Ok(Self::built(chain, "embedded", None))
    }

    /// Continuous-lag chain from proportions, mean bed lengths and optional embedded
    /// juxtapositions (Carle and Fogg).
    ///
    /// Parameters
    /// ----------
    /// proportions : array_like
    ///     Volume proportion of each category.
    /// mean_lengths : array_like
    ///     Mean bed length of each category.
    /// embedded : array_like, optional
    ///     ``(k, k)`` embedded transition probabilities; the background row and column are
    ///     filled from the others. Without it, the maximum-entropy chain.
    /// background : int
    ///     Background category; -1 means the last.
    /// symmetric : bool
    ///     Read only the upper triangle of ``embedded`` and fill the rest by symmetry.
    /// direction : {"up", "down"}
    ///
    /// Returns
    /// -------
    /// MarkovChain
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the proportions do not sum to 1, a mean length is not positive, or the
    ///     juxtapositions give no valid generator.
    #[staticmethod]
    #[pyo3(
        signature = (proportions, mean_lengths, embedded=None, *, background=-1, symmetric=false, direction="up"),
        text_signature = "(proportions, mean_lengths, embedded=None, *, background=-1, symmetric=False, direction='up')"
    )]
    fn from_geology(
        proportions: &Bound<PyAny>,
        mean_lengths: &Bound<PyAny>,
        embedded: Option<&Bound<PyAny>>,
        background: i64,
        symmetric: bool,
        direction: &str,
    ) -> PyResult<Self> {
        let p = floats(proportions, "proportions")?;
        let l = floats(mean_lengths, "mean_lengths")?;
        let e = embedded
            .filter(|o| !o.is_none())
            .map(|o| mat(o, "embedded"))
            .transpose()?;
        let background = background_index(background, p.len())?;
        let chain = MarkovChain::from_geology(
            &p,
            &l,
            e.as_ref(),
            background,
            symmetric,
            self::direction(direction)?,
        )
        .map_err(to_rust)?;
        Ok(Self::built(chain, "embedded", None))
    }

    /// "embedded", "fixed" or "continuous" (built from rates or geology).
    #[getter]
    fn kind(&self) -> &str {
        match self.chain.as_ref().map(|c| c.kind) {
            Some(Kind::Continuous) => "continuous",
            _ => &self.settings.kind,
        }
    }
    /// "up" or "down".
    #[getter]
    fn direction(&self) -> &'static str {
        direction_name(self.settings.direction)
    }
    /// Step of a fixed chain, else None.
    #[getter]
    fn step(&self) -> Option<f64> {
        match self.chain.as_ref().map(|c| c.kind) {
            Some(Kind::Fixed { step }) => Some(step),
            _ => None,
        }
    }
    /// Number of categories ``k``; InvalidInput before fitting.
    #[getter]
    fn n_categories(&self) -> PyResult<usize> {
        Ok(self.fitted()?.k())
    }
    /// Transition counts (bed transitions for embedded, step transitions for fixed), or None.
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        Ok(self.fitted()?.counts.as_ref().map(|m| mat2(py, m)))
    }
    /// ``(k, k)`` row-stochastic transition matrix (embedded or one step).
    #[getter]
    fn transition_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat2(py, self.fitted()?.transition.matrix()))
    }
    /// Generator ``Q`` with ``P(h) = exp(Q h)``, or None.
    #[getter]
    fn generator<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        Ok(self
            .fitted()?
            .generator
            .as_ref()
            .map(|q| mat2(py, q.matrix())))
    }
    /// ``(k,)`` stationary proportions.
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(py, self.fitted()?.proportions.clone()))
    }
    /// ``(k,)`` mean bed length per category, or None for a fixed chain.
    #[getter]
    fn mean_lengths<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self.fitted()?.mean_lengths.clone().map(|v| vec1(py, v)))
    }
    /// ``(k,)`` frequency of each category among beds (embedded) or steps (fixed).
    #[getter]
    fn bed_frequencies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(py, self.fitted()?.bed_frequencies.clone()))
    }
    /// Fixed chains: largest change made to ``logm(P) / step`` to make it a generator.
    #[getter]
    fn projection_distance(&self) -> PyResult<Option<f64>> {
        Ok(self.fitted()?.projection.map(|p| p.distance))
    }

    /// Transition matrix at lag ``h``.
    ///
    /// Parameters
    /// ----------
    /// h : float or array_like
    ///     Lag along the direction; whole steps for a fixed chain without a generator.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k, k)`` for a scalar, ``(n, k, k)`` for an array.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain is not fitted, a lag is invalid for it, or the stack exceeds the
    ///     available memory.
    fn transition_at<'py>(
        &self,
        py: Python<'py>,
        h: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let c = self.fitted()?;
        let (hs, scalar) = scalar_or_floats(h, "h")?;
        if scalar {
            let t = c.transition_at(hs[0]).map_err(to_rust)?;
            return Ok(mat2(py, t.matrix()).into_any());
        }
        let k = c.k();
        let need = (hs.len() as u64).saturating_mul((k * k * 8) as u64);
        transiograms_core::memory::check(need, None, "transition_at output").map_err(to_rust)?;
        let mut out = numpy::ndarray::Array3::<f64>::zeros((hs.len(), k, k));
        for (n, &h) in hs.iter().enumerate() {
            let t = c.transition_at(h).map_err(to_rust)?;
            let m = t.matrix();
            for i in 0..k {
                for j in 0..k {
                    out[(n, i, j)] = m[(i, j)];
                }
            }
        }
        Ok(numpy::IntoPyArray::into_pyarray(out, py).into_any())
    }

    /// Test of the counts against independence (quasi-independence for embedded chains).
    ///
    /// Parameters
    /// ----------
    /// method : {"g", "pearson"}
    ///
    /// Returns
    /// -------
    /// TestResult
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain has no counts (built from a matrix) or ``method`` is unknown.
    #[pyo3(signature = (*, method="g"))]
    fn independence_test(&self, method: &str) -> PyResult<PyTestResult> {
        self.fitted()?
            .independence_test(self::method(method)?)
            .map(PyTestResult)
            .map_err(to_rust)
    }

    /// Bowker test of symmetry of the counts.
    ///
    /// Returns
    /// -------
    /// TestResult
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain has no counts.
    fn symmetry_test(&self) -> PyResult<PyTestResult> {
        self.fitted()?
            .symmetry_test()
            .map(PyTestResult)
            .map_err(to_rust)
    }

    /// Hattori (1976) entropies of the chain, in bits.
    ///
    /// Returns
    /// -------
    /// Entropy
    ///     ``post`` and ``pre`` per category (``pre`` is NaN for a category never entered),
    ///     their normalized forms, the whole-chain entropy ``chain`` and its maximum ``max``.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain is not fitted.
    fn entropy(&self) -> PyResult<PyEntropy> {
        self.fitted()?.entropy().map(PyEntropy).map_err(to_rust)
    }

    /// Observed minus quasi-independent embedded transition probabilities.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k, k)``.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain has no embedded counts.
    fn difference_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat2(
            py,
            &self.fitted()?.difference_matrix().map_err(to_rust)?,
        ))
    }

    /// ``F - F^T`` of the transition flux ``F``: zero for a reversible chain.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k, k)``.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain is not fitted.
    fn asymmetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat2(py, &self.fitted()?.asymmetry().map_err(to_rust)?))
    }

    /// Kemeny and Snell standard deviation of the proportions over ``n_steps`` steps.
    ///
    /// Parameters
    /// ----------
    /// n_steps : int
    ///     Steps of the chain, at least 1.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(k,)``.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain is not fitted or ``n_steps`` is 0.
    fn proportion_std<'py>(
        &self,
        py: Python<'py>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(
            py,
            self.fitted()?.proportion_std(n_steps).map_err(to_rust)?,
        ))
    }

    /// Seeded simulation over a length along the direction.
    ///
    /// Parameters
    /// ----------
    /// length : float
    ///     Fixed chains: a multiple of ``step``. Chains with a generator: any positive length.
    /// n : int
    ///     Number of realizations.
    /// seed : int
    /// hard_from, hard_to, hard_codes : array_like, optional
    ///     Conditioning intervals as lengths along the direction from 0; a point has
    ///     ``from == to``. Fixed chains freeze the cells whose centres an interval covers, and
    ///     the cell that contains a point.
    /// start : int, "stationary" or None
    ///     Category at position 0, or a draw from the stationary law (None).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or dict
    ///     Fixed chains: int64 ``(n, length / step)``, column ``i`` the ``i``-th step along the
    ///     direction (for "up", column 0 is the deepest; reverse the columns for top-down rows).
    ///     Chains with a generator: a dict of arrays ``realization``, ``from``, ``to`` and
    ///     ``code``, one entry per bed, positions along the direction.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain is not fitted, ``length`` is not a positive multiple of a fixed
    ///     step, the conditioning contradicts itself, or ``seed`` is out of range.
    #[pyo3(signature = (length, *, n=1, seed=0, hard_from=None, hard_to=None, hard_codes=None, start=None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        length: f64,
        n: usize,
        seed: i128,
        hard_from: Option<Bound<'py, PyAny>>,
        hard_to: Option<Bound<'py, PyAny>>,
        hard_codes: Option<Bound<'py, PyAny>>,
        start: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let seed = crate::args::seed(seed)?;
        let c = self.fitted()?;
        let mut hard = Self::hard(hard_from.as_ref(), hard_to.as_ref(), hard_codes.as_ref())?;
        let start = match &start {
            Some(s) if !s.is_none() => Self::start(s)?,
            _ => Start::Stationary,
        };
        let target = match c.kind {
            Kind::Fixed { step } => {
                let steps = (length / step).round();
                if !(length > 0.0 && (length - steps * step).abs() <= 1e-9 * step) {
                    return Err(invalid(format!(
                        "length {length} must be a positive multiple of the step {step}"
                    )));
                }
                let (mut from, mut to, mut keep) = (vec![], vec![], vec![]);
                for i in 0..hard.codes.len() {
                    let (a, b) = if hard.from[i] == hard.to[i] {
                        let c = (hard.from[i] / step).floor().clamp(0.0, steps - 1.0);
                        (c, c)
                    } else {
                        (
                            (hard.from[i] / step - 0.5).ceil().max(0.0),
                            (hard.to[i] / step - 0.5).floor().min(steps - 1.0),
                        )
                    };
                    if a <= b {
                        from.push(a);
                        to.push(b);
                        keep.push(hard.codes[i]);
                    }
                }
                hard = Hard {
                    from,
                    to,
                    codes: keep,
                };
                Target::Steps(steps as usize)
            }
            _ => Target::Length(length),
        };
        let out = py
            .detach(|| c.simulate(target, &hard, start, n, seed))
            .map_err(to_rust)?;
        match out {
            Simulated::Codes(rows) => codes2(py, &rows),
            Simulated::Beds(beds) => beds_dict(py, &beds),
        }
    }

    /// Seeded simulation of a number of beds with an embedded transition matrix.
    ///
    /// Parameters
    /// ----------
    /// n_beds : int
    /// n : int
    /// seed : int
    /// hard_from, hard_to, hard_codes : array_like, optional
    ///     Conditioning as bed indices (inclusive).
    /// start : int, "stationary" or None
    ///     Category of the first bed, or a draw from the stationary law (None).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     int64 ``(n, n_beds)``, beds in order along the direction.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the chain has no embedded matrix, the conditioning contradicts itself, or
    ///     ``seed`` is out of range.
    #[pyo3(signature = (n_beds, *, n=1, seed=0, hard_from=None, hard_to=None, hard_codes=None, start=None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate_beds<'py>(
        &self,
        py: Python<'py>,
        n_beds: usize,
        n: usize,
        seed: i128,
        hard_from: Option<Bound<'py, PyAny>>,
        hard_to: Option<Bound<'py, PyAny>>,
        hard_codes: Option<Bound<'py, PyAny>>,
        start: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let seed = crate::args::seed(seed)?;
        let c = self.fitted()?;
        let hard = Self::hard(hard_from.as_ref(), hard_to.as_ref(), hard_codes.as_ref())?;
        let start = match &start {
            Some(s) if !s.is_none() => Self::start(s)?,
            _ => Start::Stationary,
        };
        let out = py
            .detach(|| c.simulate(Target::Beds(n_beds), &hard, start, n, seed))
            .map_err(to_rust)?;
        match out {
            Simulated::Codes(rows) => codes2(py, &rows),
            Simulated::Beds(beds) => beds_dict(py, &beds),
        }
    }

    /// JSON text of this object; ``MarkovChain.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(self)
    }

    /// MarkovChain from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// MarkovChain
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not MarkovChain JSON or describes an invalid MarkovChain.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        unjson(text, "MarkovChain")
    }

    fn __repr__(&self) -> String {
        match &self.chain {
            Some(c) => format!(
                "MarkovChain(kind={:?}, direction={:?}, n_categories={})",
                self.kind(),
                self.direction(),
                c.k()
            ),
            None => format!(
                "MarkovChain(kind={:?}, direction={:?}, unfitted)",
                self.settings.kind,
                self.direction()
            ),
        }
    }
}

fn beds_dict<'py>(
    py: Python<'py>,
    beds: &[(Vec<f64>, Vec<f64>, Vec<Code>)],
) -> PyResult<Bound<'py, PyAny>> {
    let (mut r, mut f, mut t, mut c): (Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>) =
        (vec![], vec![], vec![], vec![]);
    for (i, (from, to, codes)) in beds.iter().enumerate() {
        r.extend(std::iter::repeat_n(i as i64, codes.len()));
        f.extend(from);
        t.extend(to);
        c.extend(codes.iter().map(|&c| c as i64));
    }
    let d = PyDict::new(py);
    d.set_item("realization", PyArray1::from_vec(py, r))?;
    d.set_item("from", PyArray1::from_vec(py, f))?;
    d.set_item("to", PyArray1::from_vec(py, t))?;
    d.set_item("code", PyArray1::from_vec(py, c))?;
    Ok(d.into_any())
}

pub fn background_index(background: i64, k: usize) -> PyResult<usize> {
    let b = if background < 0 {
        k as i64 + background
    } else {
        background
    };
    if b < 0 || b as usize >= k {
        return Err(invalid(format!(
            "background {background} is not a category of {k}"
        )));
    }
    Ok(b as usize)
}

/// Bed statistics per category.
///
/// Attributes
/// ----------
/// n_beds, n_open : numpy.ndarray
///     Beds per category, and how many of them are cut by the end of a run (censored).
/// total, mean_length : numpy.ndarray
///     Total thickness, and thickness over closed beds (the censored exponential MLE).
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> s = tg.sojourn(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> s.n_beds.shape, s.mean_length.shape
/// ((3,), (3,))
/// >>> thickness, is_open = s.thickness(0)
/// >>> thickness.shape == is_open.shape
/// True
/// >>> at, survival = s.survival(0)
/// >>> bool(np.all(np.diff(survival) <= 0))
/// True
/// >>> s.exponential_test(0, n_bootstrap=99, seed=1).name
/// 'exponential'
/// >>> tg.Sojourn.from_json(s.to_json()).to_json() == s.to_json()
/// True
#[pyclass(module = "transiograms", name = "Sojourn", frozen)]
pub struct PySojourn(pub transiograms_chain::Sojourn);

#[pymethods]
impl PySojourn {
    /// ``(k,)`` beds per category.
    #[getter]
    fn n_beds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.0.n_beds.iter().map(|&n| n as i64).collect())
    }
    /// ``(k,)`` beds cut by the end of a run (censored).
    #[getter]
    fn n_open<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.0.n_open.iter().map(|&n| n as i64).collect())
    }
    /// ``(k,)`` total thickness per category.
    #[getter]
    fn total<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.total.clone())
    }
    /// ``(k,)`` total thickness divided by the number of closed beds (the censored
    /// exponential MLE).
    #[getter]
    fn mean_length<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.mean_length.clone())
    }

    /// Thickness of every bed of ``code`` and whether it is open (censored).
    ///
    /// Parameters
    /// ----------
    /// code : int
    ///
    /// Returns
    /// -------
    /// thickness : numpy.ndarray
    /// open : numpy.ndarray
    ///     bool, True for a bed cut by the end of its run.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``code`` is not a category.
    fn thickness<'py>(
        &self,
        py: Python<'py>,
        code: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>)> {
        self.check(code)?;
        Ok((
            vec1(py, self.0.thickness[code].clone()),
            PyArray1::from_vec(py, self.0.open[code].clone()),
        ))
    }

    /// Kaplan-Meier survival of the thickness of ``code``, open beds censored.
    ///
    /// Parameters
    /// ----------
    /// code : int
    ///
    /// Returns
    /// -------
    /// thickness : numpy.ndarray
    ///     Distinct event thicknesses, ascending.
    /// survival : numpy.ndarray
    ///     Survival just after each.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``code`` is not a category.
    fn survival<'py>(
        &self,
        py: Python<'py>,
        code: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        self.check(code)?;
        let (t, s) = self.0.survival(code as Code);
        Ok((vec1(py, t), vec1(py, s)))
    }

    /// Kolmogorov-Smirnov test of exponential thickness for ``code``, p value by seeded
    /// parametric bootstrap.
    ///
    /// Parameters
    /// ----------
    /// code : int
    /// n_bootstrap : int
    ///     Bootstrap samples.
    /// seed : int
    ///     Seed in ``0..2**64-1``.
    ///
    /// Returns
    /// -------
    /// TestResult
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``code`` is not a category, it has no closed bed, or ``seed`` is out of range.
    #[pyo3(signature = (code, *, n_bootstrap=999, seed=0))]
    fn exponential_test(
        &self,
        py: Python,
        code: usize,
        n_bootstrap: usize,
        seed: i128,
    ) -> PyResult<PyTestResult> {
        let seed = crate::args::seed(seed)?;
        self.check(code)?;
        py.detach(|| self.0.exponential_test(code as Code, n_bootstrap, seed))
            .map(PyTestResult)
            .map_err(to_rust)
    }

    /// JSON text of this object; ``Sojourn.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    /// Sojourn from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Sojourn
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not Sojourn JSON or describes an invalid Sojourn.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let s: transiograms_chain::Sojourn = unjson(text, "Sojourn")?;
        let m = s.n_beds.len();
        let lengths = [
            s.n_open.len(),
            s.total.len(),
            s.mean_length.len(),
            s.thickness.len(),
            s.open.len(),
        ];
        if lengths.iter().any(|&l| l != m) {
            return Err(invalid(format!(
                "Sojourn JSON needs {m} entries in every field, as in n_beds"
            )));
        }
        if let Some(i) = (0..m).find(|&i| s.thickness[i].len() != s.open[i].len()) {
            return Err(invalid(format!(
                "Sojourn JSON category {i} has {} thicknesses but {} open flags",
                s.thickness[i].len(),
                s.open[i].len()
            )));
        }
        Ok(Self(s))
    }
    fn __eq__(&self, other: PyRef<Self>) -> PyResult<bool> {
        same_json(&self.0, &other.0)
    }
    fn __repr__(&self) -> String {
        format!("Sojourn(n_beds={:?})", self.0.n_beds)
    }
}

impl PySojourn {
    fn check(&self, code: usize) -> PyResult<()> {
        let k = self.0.n_beds.len();
        if code >= k {
            return Err(invalid(format!("code {code} is not one of {k} categories")));
        }
        Ok(())
    }
}

fn options(kind: &str, direction: &str, gap: f64, sampled: Option<f64>) -> PyResult<FitOptions> {
    Ok(FitOptions {
        kind: kind_of(kind_name(kind)?, None, sampled)?,
        direction: self::direction(direction)?,
        gap,
        ..FitOptions::default()
    })
}

/// Test of Markov order ``order`` against ``order - 1``.
///
/// Parameters
/// ----------
/// codes : array_like or str
/// order : int
///     Fixed chains: any order >= 1 with no more histories (``m ** order``) than
///     transitions; a higher order raises ``InvalidInput``. Embedded chains: 1
///     (quasi-independence) or 2.
/// kind : {"fixed", "embedded"}
/// direction : {"up", "down"}
/// method : {"g", "pearson"}
/// holes, from_, to, depth, step, gap, categories, data
///     The shared sequence arguments, as in ``MarkovChain.fit``.
///
/// Returns
/// -------
/// TestResult
///
/// Raises
/// ------
/// InvalidInput
///     When a code is out of range, the sequence arguments disagree in length, a named column
///     is missing from ``data``, or an option is unknown.
///
/// Examples
/// --------
/// >>> codes = tg.MarkovChain.from_matrix([[0.9, 0.1], [0.2, 0.8]]).simulate(2000, seed=1)[0]
/// >>> r = tg.order_test(codes, order=2)
/// >>> r.name, r.df
/// ('order 2 vs 1', 2)
/// >>> bool(r.p_value > 0.01)
/// True
#[pyfunction]
#[pyo3(signature = (codes, *, order=2, kind="fixed", direction="up", method="g", holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn order_test<'py>(
    py: Python<'py>,
    codes: Bound<'py, PyAny>,
    order: usize,
    kind: &str,
    direction: &str,
    method: &str,
    holes: Option<Bound<'py, PyAny>>,
    from_: Option<Bound<'py, PyAny>>,
    to: Option<Bound<'py, PyAny>>,
    depth: Option<Bound<'py, PyAny>>,
    step: Option<f64>,
    gap: f64,
    categories: Option<Bound<'py, PyAny>>,
    data: Option<Bound<'py, PyAny>>,
) -> PyResult<PyTestResult> {
    let (seqs, sampled) = sequence_set_step(SequenceArgs {
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
    let opts = options(kind, direction, gap, sampled)?;
    let m = self::method(method)?;
    py.detach(|| transiograms_chain::order_test(&seqs, &opts, order, m))
        .map(PyTestResult)
        .map_err(to_rust)
}

/// AIC and BIC of fixed-step chains of order 0 to ``max_order``, all fitted on the same
/// transitions.
///
/// Parameters
/// ----------
/// codes : array_like or str
/// max_order : int
///     Highest order fitted; the first ``max_order`` states of every run are dropped so that
///     every order uses the same transitions.
/// direction : {"up", "down"}
/// holes, from_, to, depth, step, gap, categories, data
///     The shared sequence arguments, as in ``MarkovChain.fit``.
///
/// Returns
/// -------
/// dict
///     ``order``, ``log_likelihood``, ``n_params``, ``aic``, ``bic`` and ``n_transitions`` (an
///     int array of length 1).
///
/// Raises
/// ------
/// InvalidInput
///     When a code is out of range, the sequence arguments disagree in length, a named column
///     is missing from ``data``, or an option is unknown.
///
/// Examples
/// --------
/// >>> codes = tg.MarkovChain.from_matrix([[0.9, 0.1], [0.2, 0.8]]).simulate(2000, seed=1)[0]
/// >>> table = tg.select_order(codes, max_order=2)
/// >>> table["order"]
/// array([0, 1, 2])
/// >>> int(table["order"][np.argmin(table["bic"])])
/// 1
#[pyfunction]
#[pyo3(signature = (codes, *, max_order=3, direction="up", holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn select_order<'py>(
    py: Python<'py>,
    codes: Bound<'py, PyAny>,
    max_order: usize,
    direction: &str,
    holes: Option<Bound<'py, PyAny>>,
    from_: Option<Bound<'py, PyAny>>,
    to: Option<Bound<'py, PyAny>>,
    depth: Option<Bound<'py, PyAny>>,
    step: Option<f64>,
    gap: f64,
    categories: Option<Bound<'py, PyAny>>,
    data: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (seqs, sampled) = sequence_set_step(SequenceArgs {
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
    let opts = options("fixed", direction, gap, sampled)?;
    let t = py
        .detach(|| transiograms_chain::select_order(&seqs, &opts, max_order))
        .map_err(to_rust)?;
    let ints = |v: &[usize]| PyArray1::from_vec(py, v.iter().map(|&x| x as i64).collect());
    let d = PyDict::new(py);
    d.set_item("order", ints(&t.orders))?;
    d.set_item("log_likelihood", vec1(py, t.log_likelihood))?;
    d.set_item("n_params", ints(&t.n_params))?;
    d.set_item("aic", vec1(py, t.aic))?;
    d.set_item("bic", vec1(py, t.bic))?;
    d.set_item("n_transitions", ints(&[t.n_transitions]))?;
    Ok(d)
}

/// Test that groups of holes share one transition matrix.
///
/// Parameters
/// ----------
/// codes : array_like or str
/// groups : array_like or str
///     Group label per row; transitions are counted only between rows of one group.
/// kind, direction, method
///     As in ``order_test``.
/// holes, from_, to, depth, step, gap, categories, data
///     The shared sequence arguments.
///
/// Returns
/// -------
/// TestResult
///
/// Raises
/// ------
/// InvalidInput
///     When a code is out of range, the sequence arguments disagree in length, a named column
///     is missing from ``data``, or an option is unknown.
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> groups = np.where(np.isin(d["HOLE_ID"], ["DH01", "DH02", "DH03", "DH04", "DH05", "DH06"]), "A", "B")
/// >>> r = tg.homogeneity_test(d["CODE"], groups, holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> r.name
/// 'homogeneity'
/// >>> bool(r.p_value > 0.01)
/// True
#[pyfunction]
#[pyo3(signature = (codes, groups, *, kind="embedded", direction="up", method="g", holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn homogeneity_test<'py>(
    py: Python<'py>,
    codes: Bound<'py, PyAny>,
    groups: Bound<'py, PyAny>,
    kind: &str,
    direction: &str,
    method: &str,
    holes: Option<Bound<'py, PyAny>>,
    from_: Option<Bound<'py, PyAny>>,
    to: Option<Bound<'py, PyAny>>,
    depth: Option<Bound<'py, PyAny>>,
    step: Option<f64>,
    gap: f64,
    categories: Option<Bound<'py, PyAny>>,
    data: Option<Bound<'py, PyAny>>,
) -> PyResult<PyTestResult> {
    let np = py.import("numpy")?;
    let d = data.as_ref();
    let col = |o: &Bound<'py, PyAny>, what: &str| crate::args::column(d, o, what);
    let all = crate::args::codes(&col(&codes, "codes")?, "codes")?;
    let n = all.len();
    let labels = hole_ids(Some(&col(&groups, "groups")?), n)?;
    let k = crate::args::categories_k(categories.as_ref(), &all)?;
    let mut order: Vec<String> = Vec::new();
    for g in &labels {
        if !order.contains(g) {
            order.push(g.clone());
        }
    }
    let take = |o: &Option<Bound<'py, PyAny>>,
                rows: &Bound<'py, PyAny>,
                what: &str|
     -> PyResult<Option<Bound<'py, PyAny>>> {
        match o.as_ref().filter(|o| !o.is_none()) {
            None => Ok(None),
            Some(o) => Ok(Some(
                np.call_method1("asarray", (col(o, what)?,))?
                    .get_item(rows)?,
            )),
        }
    };
    let mut sets: Vec<SequenceSet> = Vec::new();
    let mut sampled = None;
    for g in &order {
        let rows: Vec<usize> = (0..n).filter(|&i| &labels[i] == g).collect();
        let rows = PyArray1::from_vec(py, rows.iter().map(|&r| r as i64).collect()).into_any();
        let (set, s) = sequence_set_step(SequenceArgs {
            codes: np
                .call_method1("asarray", (col(&codes, "codes")?,))?
                .get_item(&rows)?,
            holes: take(&holes, &rows, "holes")?,
            from_: take(&from_, &rows, "from_")?,
            to: take(&to, &rows, "to")?,
            depth: take(&depth, &rows, "depth")?,
            step,
            gap,
            categories: Some(k.into_pyobject(py)?.into_any()),
            data: None,
        })?;
        sampled = sampled.or(s);
        sets.push(set);
    }
    let opts = options(kind, direction, gap, sampled)?;
    let m = self::method(method)?;
    py.detach(|| transiograms_chain::homogeneity_test(&sets, &opts, m))
        .map(PyTestResult)
        .map_err(to_rust)
}

/// Bed thickness statistics per category.
///
/// Parameters
/// ----------
/// codes : array_like or str
/// direction : {"up", "down"}
/// holes, from_, to, depth, step, gap, categories, data
///     The shared sequence arguments.
///
/// Returns
/// -------
/// Sojourn
///
/// Raises
/// ------
/// InvalidInput
///     When a code is out of range, the sequence arguments disagree in length, a named column
///     is missing from ``data``, or an option is unknown.
///
/// Examples
/// --------
/// >>> d = tg.datasets.varved_sequence()
/// >>> s = tg.sojourn(d["CODE"], holes=d["HOLE_ID"], from_=d["FROM"], to=d["TO"])
/// >>> int(s.n_beds.sum()) > 0
/// True
#[pyfunction]
#[pyo3(signature = (codes, *, direction="up", holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn sojourn<'py>(
    codes: Bound<'py, PyAny>,
    direction: &str,
    holes: Option<Bound<'py, PyAny>>,
    from_: Option<Bound<'py, PyAny>>,
    to: Option<Bound<'py, PyAny>>,
    depth: Option<Bound<'py, PyAny>>,
    step: Option<f64>,
    gap: f64,
    categories: Option<Bound<'py, PyAny>>,
    data: Option<Bound<'py, PyAny>>,
) -> PyResult<PySojourn> {
    let (seqs, _) = sequence_set_step(SequenceArgs {
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
    Ok(PySojourn(transiograms_chain::sojourn(
        &seqs,
        self::direction(direction)?,
        gap,
    )))
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTestResult>()?;
    m.add_class::<PyEntropy>()?;
    m.add_class::<PyMarkovChain>()?;
    m.add_class::<PySojourn>()?;
    m.add_function(wrap_pyfunction!(order_test, m)?)?;
    m.add_function(wrap_pyfunction!(select_order, m)?)?;
    m.add_function(wrap_pyfunction!(homogeneity_test, m)?)?;
    m.add_function(wrap_pyfunction!(sojourn, m)?)?;
    Ok(())
}
