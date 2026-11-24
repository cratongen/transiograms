"""Tests for the __minetypes__ protocol v1 producer and consumer contract."""

import numpy as np
import pytest
import transiograms as tg

COMMON_KEYS = {"protocol", "kind", "columns", "units", "categories", "crs", "length_unit"}


class ProtoWrapper:
    """Wraps a raw protocol dict in an object implementing __minetypes__()."""

    def __init__(self, d: dict):
        self._d = d

    def __minetypes__(self) -> dict:
        return self._d


def ro(arr: np.ndarray) -> np.ndarray:
    """Marks a C-contiguous array read-only as required by protocol v1."""
    arr = np.ascontiguousarray(arr)
    arr.flags.writeable = False
    return arr


def make_category_entry(names=("sand", "silt", "clay"), colors=None, mapping=None, other=None):
    return {
        "names": list(names),
        "colors": list(colors) if colors is not None else None,
        "mapping": dict(mapping) if mapping is not None else {},
        "other": other,
    }


def make_categories_proto(names=("sand", "silt", "clay"), colors=("#e6c35c", "#a0a0a0", "#6b4f2a")):
    d = {
        "protocol": 1,
        "kind": "categories",
        "columns": {},
        "units": {},
        "categories": {},
        "crs": None,
        "length_unit": None,
    }
    d.update(make_category_entry(names=names, colors=colors))
    return d


def make_grid_proto(
    origin=(0.0, 0.0, 0.0),
    size=(10.0, 10.0, 1.0),
    count=(4, 3, 2),
    rotation=(0.0, 0.0, 0.0),
    columns=None,
    categories=None,
    crs="EPSG:31982",
    length_unit="m",
):
    return {
        "protocol": 1,
        "kind": "grid",
        "origin": origin,
        "size": size,
        "count": count,
        "rotation": rotation,
        "columns": columns if columns is not None else {},
        "units": {},
        "categories": categories if categories is not None else {},
        "crs": crs,
        "length_unit": length_unit,
    }


def make_block_model_proto(
    layout="regular",
    origin=(0.0, 0.0, 0.0),
    size=(10.0, 10.0, 1.0),
    count=(4, 3, 2),
    rotation=(0.0, 0.0, 0.0),
    index=None,
    parent=None,
    offset=None,
    scale=None,
    subgrid=None,
    columns=None,
    categories=None,
    crs="EPSG:31982",
    length_unit="m",
):
    return {
        "protocol": 1,
        "kind": "block_model",
        "origin": origin,
        "size": size,
        "count": count,
        "rotation": rotation,
        "layout": layout,
        "index": index,
        "parent": parent,
        "offset": offset,
        "scale": scale,
        "subgrid": subgrid,
        "columns": columns if columns is not None else {},
        "units": {},
        "categories": categories if categories is not None else {},
        "crs": crs,
        "length_unit": length_unit,
    }


def make_intervals_proto(
    holes=("DH01", "DH02"),
    hole_index=(0, 0, 0, 1, 1, 1),
    from_depth=(0.0, 1.0, 2.5, 0.0, 1.5, 3.0),
    to_depth=(1.0, 2.5, 4.0, 1.5, 3.0, 5.0),
    codes=(0, 1, 2, 1, 2, 0),
    col_name="lith",
    names=("sand", "silt", "clay"),
):
    code_arr = ro(np.asarray(codes, dtype=np.int32))
    return {
        "protocol": 1,
        "kind": "intervals",
        "holes": list(holes),
        "hole_index": ro(np.asarray(hole_index, dtype=np.int32)),
        "from_depth": ro(np.asarray(from_depth, dtype=np.float64)),
        "to_depth": ro(np.asarray(to_depth, dtype=np.float64)),
        "columns": {col_name: code_arr},
        "units": {},
        "categories": {col_name: make_category_entry(names=names)},
        "crs": "EPSG:31982",
        "length_unit": "m",
    }


def make_drillholes_proto(with_primary=True, with_table=False):
    iv = make_intervals_proto()
    nested_iv = {
        "hole_index": iv["hole_index"],
        "from_depth": iv["from_depth"],
        "to_depth": iv["to_depth"],
        "columns": iv["columns"],
        "units": iv["units"],
        "categories": iv["categories"],
    }
    return {
        "protocol": 1,
        "kind": "drillholes",
        "holes": ["DH01", "DH02"],
        "coords": ro(np.array([[100.0, 200.0, 50.0], [120.0, 200.0, 50.0]], dtype=np.float64)),
        "survey": {
            "hole_index": ro(np.array([0, 0, 1, 1], dtype=np.int32)),
            "depth": ro(np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)),
            "azimuth": ro(np.array([0.0, 0.0, 90.0, 90.0], dtype=np.float64)),
            "dip": ro(np.array([-90.0, -90.0, -85.0, -85.0], dtype=np.float64)),
        },
        "intervals": nested_iv if with_primary else None,
        "tables": {"lithology": nested_iv} if with_table else {},
        "path_coords": ro(
            np.array(
                [
                    [100.0, 200.0, 50.0],
                    [100.0, 200.0, 40.0],
                    [120.0, 200.0, 50.0],
                    [120.87, 200.0, 40.04],
                ],
                dtype=np.float64,
            )
        ),
        "path_depths": ro(np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)),
        "path_offsets": ro(np.array([0, 2, 4], dtype=np.uint32)),
        "method": "minimum_curvature",
        "columns": {"zone": ro(np.array(["north", "south"]))},
        "units": {},
        "categories": {},
        "crs": "EPSG:31982",
        "length_unit": "m",
    }


def make_composites_proto():
    hi = ro(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
    fd = ro(np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64))
    td = ro(np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64))
    length = ro(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64))
    coords = ro(
        np.array(
            [
                [5.0, 5.0, 2.5],
                [5.0, 5.0, 1.5],
                [5.0, 5.0, 0.5],
                [15.0, 5.0, 2.5],
                [15.0, 5.0, 1.5],
                [15.0, 5.0, 0.5],
            ],
            dtype=np.float64,
        )
    )
    facies = ro(np.array([0, 1, 2, 1, 2, 0], dtype=np.int32))
    au = ro(np.array([1.2, 0.8, np.nan, 0.5, 1.9, 0.3], dtype=np.float64))
    au_len = ro(np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64))
    domain = ro(np.array([0, 0, 1, 0, 1, 1], dtype=np.int32))
    return {
        "protocol": 1,
        "kind": "composites",
        "holes": ["DH01", "DH02"],
        "coords": coords,
        "hole_index": hi,
        "from_depth": fd,
        "to_depth": td,
        "length": length,
        "domain": domain,
        "columns": {"facies": facies, "au": au, "au_length": au_len},
        "units": {"au": "g/t"},
        "categories": {
            "facies": make_category_entry(names=("sand", "silt", "clay")),
            "domain": make_category_entry(names=("oxide", "fresh")),
        },
        "crs": "EPSG:31982",
        "length_unit": "m",
    }


def make_point_set_proto():
    coords = ro(
        np.array(
            [
                [5.0, 5.0, 0.5],
                [15.0, 5.0, 0.5],
                [25.0, 15.0, 1.5],
                [35.0, 25.0, 1.5],
            ],
            dtype=np.float64,
        )
    )
    codes = ro(np.array([0, 1, 2, 0], dtype=np.int32))
    return {
        "protocol": 1,
        "kind": "point_set",
        "coords": coords,
        "columns": {"facies": codes},
        "units": {},
        "categories": {
            "facies": make_category_entry(
                names=("sand", "silt", "clay"),
                colors=("#e6c35c", "#a0a0a0", "#6b4f2a"),
            )
        },
        "crs": "EPSG:31982",
        "length_unit": "m",
    }


def test_categories_producer_and_consumer_contract():
    cats = tg.Categories(["sand", "silt", "clay"], colors=["#e6c35c", "#a0a0a0", "#6b4f2a"])
    d = cats.__minetypes__()
    assert COMMON_KEYS <= set(d)
    assert d["protocol"] == 1
    assert d["kind"] == "categories"
    assert d["names"] == ["sand", "silt", "clay"]
    assert d["colors"] == ["#e6c35c", "#a0a0a0", "#6b4f2a"]
    assert d["mapping"] == {}
    assert d["other"] is None
    assert d["columns"] == {} and d["units"] == {} and d["categories"] == {}
    assert d["crs"] is None and d["length_unit"] is None

    no_col = tg.Categories(["a", "b"]).__minetypes__()
    assert no_col["colors"] is None

    rebuilt = tg.Categories(ProtoWrapper(make_categories_proto()))
    assert rebuilt == cats
    assert tg.Categories(d) == cats


def test_grid_producer_and_consumer_contract():
    grid = tg.Grid((100.0, 200.0, 10.0), (5.0, 5.0, 2.0), (4, 3, 2))
    d = grid.__minetypes__()
    assert set(d) == COMMON_KEYS | {"origin", "size", "count", "rotation"}
    assert "coords" not in d
    assert d["protocol"] == 1
    assert d["kind"] == "grid"
    assert d["origin"] == (100.0, 200.0, 10.0)
    assert d["size"] == (5.0, 5.0, 2.0)
    assert d["count"] == (4, 3, 2)
    assert d["rotation"] == (0.0, 0.0, 0.0)
    assert d["columns"] == {} and d["units"] == {} and d["categories"] == {}
    assert d["crs"] is None and d["length_unit"] is None

    # Rule 2: building the dict allocates nothing proportional to cell count.
    huge = tg.Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (100_000, 100_000, 100_000))
    hd = huge.__minetypes__()
    assert hd["count"] == (100_000, 100_000, 100_000) and "coords" not in hd

    assert tg.Grid(d) == grid
    assert tg.Grid(ProtoWrapper(d)) == grid


def test_realizations_producer_int32_codes_and_metadata():
    model = tg.TransitionModel.from_geology(
        [0.5, 0.3, 0.2], [[20.0, 15.0, 10.0], [20.0, 15.0, 10.0], [4.0, 3.0, 2.0]]
    )
    bm = make_block_model_proto(
        layout="masked",
        origin=(0.0, 0.0, 0.0),
        size=(10.0, 10.0, 1.0),
        count=(4, 3, 2),
        index=ro(np.arange(1, 24, dtype=np.uint64)),
        crs="epsg:31982",
        length_unit="m",
    )
    pts = make_point_set_proto()
    pts["crs"] = "EPSG:31982"
    real = tg.mcrf_simulate(model, ProtoWrapper(bm), ProtoWrapper(pts), n=2, seed=7)
    d = real.__minetypes__()

    assert set(d) == COMMON_KEYS | {"origin", "size", "count", "rotation"}
    assert d["protocol"] == 1
    assert d["kind"] == "grid"
    assert d["origin"] == (0.0, 0.0, 0.0)
    assert d["size"] == (10.0, 10.0, 1.0)
    assert d["count"] == (4, 3, 2)
    assert d["rotation"] == (0.0, 0.0, 0.0)
    assert d["crs"] == "epsg:31982"
    assert d["length_unit"] == "m"
    assert list(d["columns"]) == ["realization_0", "realization_1"]
    assert list(d["categories"]) == ["realization_0", "realization_1"]

    for i, col_name in enumerate(["realization_0", "realization_1"]):
        arr = d["columns"][col_name]
        assert arr.dtype == np.int32
        assert arr.shape == (24,)
        assert arr.flags.c_contiguous
        assert arr.dtype.byteorder in ("=", "|")
        assert not arr.flags.writeable
        # Cell 0 was excluded by the masked block_model index, so it carries -1.
        assert arr[0] == -1
        assert np.all((arr[1:] >= 0) & (arr[1:] < 3))
        np.testing.assert_array_equal(arr, real.realizations[i].astype(np.int32))
        assert not np.shares_memory(arr, real.realizations)

        cat = d["categories"][col_name]
        assert cat["names"] == ["sand", "silt", "clay"]
        assert cat["colors"] == ["#e6c35c", "#a0a0a0", "#6b4f2a"]
        assert cat["mapping"] == {}
        assert cat["other"] is None

    # Realizations can be consumed directly as a grid protocol object in grid_transiogram.
    exp = tg.grid_transiogram("realization_0", real, offset=(1, 0, 0), n_lags=2)
    assert exp.probabilities.shape == (2, 3, 3)


def test_sequence_routines_consume_intervals_drillholes_and_composites():
    iv = ProtoWrapper(make_intervals_proto())
    dh_primary = ProtoWrapper(make_drillholes_proto(with_primary=True, with_table=False))
    dh_table = ProtoWrapper(make_drillholes_proto(with_primary=False, with_table=True))
    comp = ProtoWrapper(make_composites_proto())

    # Fit MarkovChain with no column plumbing in user code.
    c_iv = tg.MarkovChain(pseudocount=0.5).fit(iv)
    c_dh1 = tg.MarkovChain(pseudocount=0.5).fit(dh_primary)
    c_dh2 = tg.MarkovChain(pseudocount=0.5).fit(dh_table)
    c_explicit = tg.MarkovChain(pseudocount=0.5).fit("lith", data=dh_table)
    assert c_iv.n_categories == 3
    np.testing.assert_allclose(c_iv.transition_matrix, c_dh1.transition_matrix)
    np.testing.assert_allclose(c_iv.transition_matrix, c_dh2.transition_matrix)
    np.testing.assert_allclose(c_iv.transition_matrix, c_explicit.transition_matrix)

    c_comp = tg.MarkovChain(pseudocount=0.5).fit("facies", data=comp)
    assert c_comp.n_categories == 3

    # downhole_transiogram, sojourn, order_test, select_order, homogeneity_test
    exp = tg.downhole_transiogram(dh_primary, lags=[0.5, 1.0])
    assert exp.probabilities.shape == (2, 3, 3)
    soj = tg.sojourn(dh_primary)
    assert soj.n_beds.shape == (3,)
    ot = tg.order_test(dh_primary, order=1, kind="embedded")
    assert ot.name == "independence"
    so = tg.select_order(comp, max_order=1, step=1.0, data=None)
    assert list(so["order"]) == [0, 1]
    # homogeneity_test grouping by collar column "zone" on drillholes
    ht = tg.homogeneity_test(dh_primary, "zone")
    assert ht.name == "homogeneity"

    # HMM and HSMM consume intervals / drillholes / composites
    hmm = tg.HiddenMarkovModel(2, emission="categorical", n_symbols=3, max_iter=5, seed=1)
    hmm.fit("lith", data=dh_primary)
    pred = hmm.predict("lith", data=dh_primary)
    assert pred.shape == (6,)


def test_spatial_routines_consume_grid_block_model_point_set_and_composites():
    model = tg.TransitionModel.from_geology(
        [0.5, 0.3, 0.2], [[20.0, 15.0, 10.0], [20.0, 15.0, 10.0], [4.0, 3.0, 2.0]]
    )
    grid_p = ProtoWrapper(make_grid_proto())
    bm_reg = ProtoWrapper(make_block_model_proto(layout="regular"))
    active_idx = ro(np.array([0, 1, 2, 3, 4, 5, 14, 15, 22, 23], dtype=np.uint64))
    bm_masked = ProtoWrapper(make_block_model_proto(layout="masked", index=active_idx))
    pts = ProtoWrapper(make_point_set_proto())
    comp = ProtoWrapper(make_composites_proto())

    # Grid.snap accepts point_set and composites protocol objects
    g = tg.Grid(grid_p)
    cells_p, codes_p = g.snap(pts)
    assert cells_p.dtype == np.uint64 and codes_p.dtype == np.int64
    assert len(cells_p) == 4
    cells_c, codes_c = g.snap(comp, "facies")
    assert len(cells_c) > 0 and codes_c.dtype == np.int64

    # experimental_transiogram accepts point_set and composites (with hole_index for pairs="same_hole")
    exp_pts = tg.experimental_transiogram(pts, lag=15.0, n_lags=2)
    assert exp_pts.probabilities.shape == (2, 3, 3)
    exp_comp = tg.experimental_transiogram(comp, "facies", lag=1.0, n_lags=2, pairs="same_hole")
    assert exp_comp.probabilities.shape == (2, 3, 3)
    assert exp_comp.counts.sum() > 0

    # tp_cokriging, tprogs_simulate, mcp_predict, mcrf_simulate, fit_quality
    pred_tp = tg.tp_cokriging(model, bm_reg, pts)
    assert pred_tp.probabilities.shape == (24, 3)
    pred_mcp = tg.mcp_predict(model, bm_masked, comp, "facies")
    assert pred_mcp.probabilities.shape == (24, 3)
    assert np.all(pred_mcp.most_likely[6:14] == -1)

    sim_tp = tg.tprogs_simulate(model, grid_p, pts, n=2, seed=1, quench_iterations=1)
    assert sim_tp.realizations.shape == (2, 24)
    sim_mcrf = tg.mcrf_simulate(model, bm_masked, pts, n=2, seed=1)
    assert sim_mcrf.realizations.shape == (2, 24)
    assert np.all(sim_mcrf.realizations[:, 6:14] == -1)

    fq = tg.fit_quality(model, bm_masked, sim_mcrf, n_lags=1)
    assert fq.proportions.shape == (2, 3)

    # grid_transiogram on a masked block_model with a categorical column
    masked_codes = ro(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int32))
    bm_with_col = make_block_model_proto(
        layout="masked",
        index=active_idx,
        columns={"facies": masked_codes},
        categories={"facies": make_category_entry()},
    )
    gt = tg.grid_transiogram(ProtoWrapper(bm_with_col), offset=(1, 0, 0), n_lags=1)
    assert gt.probabilities.shape == (1, 3, 3)


def test_subblocked_block_model_validation_and_rejection_as_simulation_target():
    model = tg.TransitionModel.from_geology([0.5, 0.5], [[10.0, 10.0], [10.0, 10.0], [2.0, 2.0]])
    parent = ro(np.array([0, 0], dtype=np.uint64))
    offset = ro(np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float32))
    scale = ro(np.array([[0.5, 1.0, 1.0], [0.5, 1.0, 1.0]], dtype=np.float32))
    bm_sub = make_block_model_proto(
        layout="subblocked",
        parent=parent,
        offset=offset,
        scale=scale,
        subgrid=(2, 1, 1),
    )
    with pytest.raises(tg.InvalidInput, match="subblocked"):
        tg.mcp_predict(model, ProtoWrapper(bm_sub), [[1.0, 1.0, 0.5]], [0])

    # Misaligned subgrid is caught before the layout rejection.
    bad_sub = dict(bm_sub, subgrid=(3, 1, 1))
    with pytest.raises(tg.InvalidInput, match="subgrid"):
        tg.mcp_predict(model, ProtoWrapper(bad_sub), [[1.0, 1.0, 0.5]], [0])


def test_protocol_validation_errors_and_views():
    # View and read-only check on hand-built dict
    raw = np.array([0, 1, 2, 1, 2, 0], dtype=np.int32)
    view = ro(raw)
    assert np.shares_memory(raw, view)
    iv = make_intervals_proto(codes=view)
    assert np.shares_memory(raw, iv["columns"]["lith"])
    tg.MarkovChain(pseudocount=0.5).fit(ProtoWrapper(iv))
    np.testing.assert_array_equal(raw, [0, 1, 2, 1, 2, 0])

    # Missing common key
    for key in COMMON_KEYS:
        broken = dict(make_grid_proto())
        del broken[key]
        with pytest.raises(tg.InvalidInput, match=key):
            tg.Grid(ProtoWrapper(broken))

    # Wrong protocol version or bool
    with pytest.raises(tg.InvalidInput, match="protocol"):
        tg.Grid(ProtoWrapper(dict(make_grid_proto(), protocol=2)))
    with pytest.raises(tg.InvalidInput, match="protocol"):
        tg.Grid(ProtoWrapper(dict(make_grid_proto(), protocol=True)))

    # Unknown kind or unaccepted kind
    with pytest.raises(tg.InvalidInput, match="voxel"):
        tg.Grid(ProtoWrapper(dict(make_grid_proto(), kind="voxel")))
    with pytest.raises(tg.InvalidInput, match="categories"):
        tg.Grid(ProtoWrapper(make_categories_proto()))

    # Non-zero grid rotation rejected by axis-aligned tg.Grid
    with pytest.raises(tg.InvalidInput, match="rotation"):
        tg.Grid(ProtoWrapper(make_grid_proto(rotation=(15.0, 0.0, 0.0))))

    # Grid must not carry explicit coords
    with pytest.raises(tg.InvalidInput, match="coords"):
        tg.Grid(ProtoWrapper(dict(make_grid_proto(), coords=ro(np.zeros((24, 3))))))

    # Non-contiguous array rejected
    non_contig = np.arange(12, dtype=np.int32)[::2]
    non_contig.flags.writeable = False
    bad_iv = make_intervals_proto()
    bad_iv["columns"] = {"lith": non_contig}
    with pytest.raises(tg.InvalidInput, match="C-contiguous"):
        tg.MarkovChain().fit(ProtoWrapper(bad_iv))

    # Foreign-endian array rejected
    swapped = np.array([0, 1, 2, 1, 2, 0], dtype=">i4")
    swapped.flags.writeable = False
    bad_endian = make_intervals_proto()
    bad_endian["columns"] = {"lith": swapped}
    with pytest.raises(tg.InvalidInput, match="native-endian"):
        tg.MarkovChain().fit(ProtoWrapper(bad_endian))

    # Category code column must be int32, not int64
    int64_codes = ro(np.array([0, 1, 2, 1, 2, 0], dtype=np.int64))
    bad_dtype = make_intervals_proto()
    bad_dtype["columns"] = {"lith": int64_codes}
    with pytest.raises(tg.InvalidInput, match="int32"):
        tg.MarkovChain().fit(ProtoWrapper(bad_dtype))

    # Out-of-range category code (< -1 or >= K)
    for bad_code in (-2, 3):
        bad_c = make_intervals_proto(codes=(0, 1, bad_code, 1, 2, 0))
        with pytest.raises(tg.InvalidInput, match="lith"):
            tg.MarkovChain().fit(ProtoWrapper(bad_c))

    # Drillhole survey dip positive downward (> 90) or out of [-90, 90] rejected
    bad_dh = make_drillholes_proto()
    bad_survey = dict(bad_dh["survey"])
    bad_survey["dip"] = ro(np.array([-90.0, -90.0, 95.0, 95.0], dtype=np.float64))
    bad_dh["survey"] = bad_survey
    with pytest.raises(tg.InvalidInput, match="dip"):
        tg.MarkovChain().fit(ProtoWrapper(bad_dh))


def test_crs_and_length_unit_matching_and_errors():
    model = tg.TransitionModel.from_geology(
        [0.5, 0.3, 0.2], [[20.0, 15.0, 10.0], [20.0, 15.0, 10.0], [4.0, 3.0, 2.0]]
    )
    wkt = 'PROJCS["SIRGAS 2000 / UTM zone 22S",AUTHORITY["EPSG","31982"]]'
    g = make_grid_proto(crs="epsg:31982", length_unit=" m ")
    pts = make_point_set_proto()
    pts["crs"] = wkt
    pts["length_unit"] = "m"

    real = tg.mcrf_simulate(model, ProtoWrapper(g), ProtoWrapper(pts), n=1, seed=0)
    out = real.__minetypes__()
    assert out["crs"] == "epsg:31982"
    assert out["length_unit"] == "m"

    # Mismatched CRS raises InvalidInput
    pts_bad_crs = dict(pts, crs="EPSG:31983")
    with pytest.raises(tg.InvalidInput, match="crs mismatch"):
        tg.tp_cokriging(model, ProtoWrapper(g), ProtoWrapper(pts_bad_crs))

    # Mismatched length_unit ("m" vs "metre" or "ft") raises InvalidInput
    pts_bad_lu = dict(pts, length_unit="metre")
    with pytest.raises(tg.InvalidInput, match="length_unit mismatch"):
        tg.tp_cokriging(model, ProtoWrapper(g), ProtoWrapper(pts_bad_lu))


def test_boitata_drillholes_fits_markov_chain_without_column_plumbing():
    bt = pytest.importorskip("boitata")
    d = tg.datasets.varved_sequence(seed=0)
    holes = sorted(set(d["HOLE_ID"].tolist()))
    collars = {
        "HOLE_ID": np.array(holes),
        "X": np.arange(len(holes), dtype=float) * 10.0,
        "Y": np.zeros(len(holes), dtype=float),
        "Z": np.zeros(len(holes), dtype=float),
    }
    surveys = {
        "HOLE_ID": np.array(holes),
        "DEPTH": np.zeros(len(holes), dtype=float),
        "AZIMUTH": np.zeros(len(holes), dtype=float),
        "DIP": np.full(len(holes), 90.0, dtype=float),
    }
    names = np.asarray(d["names"], dtype=object)
    intervals = {
        "HOLE_ID": d["HOLE_ID"],
        "FROM": d["FROM"],
        "TO": d["TO"],
        "LITH": names[d["CODE"]],
    }
    dh = bt.Drillholes(collars, surveys, intervals)
    if not hasattr(dh, "__minetypes__"):
        pytest.skip("installed boitata does not yet implement __minetypes__")
    chain = tg.MarkovChain().fit(dh)
    assert chain.n_categories == 3
    np.testing.assert_allclose(chain.transition_matrix.sum(axis=1), 1.0, atol=1e-12)
