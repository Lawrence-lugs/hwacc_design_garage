"""Tests for the cnode_factory module and related Cgraph utilities.

These tests are entirely self-contained — they build synthetic ONNX models
using onnx.helper so that no model files need to be present on disk.

Covered:
- cnode_factory.get_cnode_from_onnx_node for standard Conv, depthwise Conv,
  Gemm, and simple (no-initializer) node types
- Depthwise Conv bug-fix: verify that both the per-channel conv_nodes AND the
  cat_node are returned (previously only convs was returned, catter was dropped)
- Cgraph.get_matrix_dict()
"""

from __future__ import annotations

import numpy as np
import pytest
import onnx
from onnx import helper, numpy_helper, TensorProto

from hwacctools.comp_graph import cnode_factory, cnodes
from hwacctools.comp_graph.cgraph import Cgraph


# ---------------------------------------------------------------------------
# Synthetic ONNX model builders
# ---------------------------------------------------------------------------

def _make_model(node, initializers, opset=17):
    """Wrap a single node + initializers into a minimal ModelProto."""
    graph = helper.make_graph(
        nodes=[node],
        name="test_graph",
        inputs=[],
        outputs=[helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def make_conv_model(K=4, C=3, kH=3, kW=3, group=1, strides=1, pads=1):
    """Return (model, node) for a single Conv op.

    Attributes are ordered alphabetically (dilations, group, kernel_shape,
    pads, strides) so that node.attribute[1] is ``group``, matching the
    assumption in get_cnode_from_onnx_node.
    """
    C_per_group = C // group
    kernel = np.random.randn(K, C_per_group, kH, kW).astype(np.float32)
    bias = np.zeros(K, dtype=np.float32)

    inits = [
        numpy_helper.from_array(kernel, name="w"),
        numpy_helper.from_array(bias, name="b"),
    ]
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        dilations=[1, 1],
        group=group,
        kernel_shape=[kH, kW],
        pads=[pads, pads, pads, pads],
        strides=[strides, strides],
    )
    return _make_model(node, inits), node


def make_gemm_model(in_features=10, out_features=5):
    """Return (model, node) for a single Gemm op."""
    B = np.random.randn(out_features, in_features).astype(np.float32)
    C = np.zeros(out_features, dtype=np.float32)

    inits = [
        numpy_helper.from_array(B, name="B"),
        numpy_helper.from_array(C, name="C"),
    ]
    node = helper.make_node(
        "Gemm",
        inputs=["x", "B", "C"],
        outputs=["y"],
    )
    return _make_model(node, inits), node


def make_simple_node(op_type, inputs, outputs, **attrs):
    """Return (model, node) for a single node with no initializers."""
    node = helper.make_node(op_type, inputs=inputs, outputs=outputs, **attrs)
    graph = helper.make_graph(
        [node], "g",
        inputs=[],
        outputs=[helper.make_tensor_value_info(outputs[0], TensorProto.FLOAT, None)],
    )
    return helper.make_model(graph), node


# ---------------------------------------------------------------------------
# Tests: import paths
# ---------------------------------------------------------------------------

class TestImportPaths:
    """The factory function must be reachable from cnode_factory."""

    def test_import_from_cnode_factory(self):
        from hwacctools.comp_graph.cnode_factory import get_cnode_from_onnx_node
        assert callable(get_cnode_from_onnx_node)

    def test_cnode_factory_module_attribute(self):
        assert hasattr(cnode_factory, "get_cnode_from_onnx_node")
        assert callable(cnode_factory.get_cnode_from_onnx_node)


# ---------------------------------------------------------------------------
# Tests: standard (non-depthwise) Conv
# ---------------------------------------------------------------------------

class TestStandardConv:
    """Standard Conv (group=1) should return a single conv_node."""

    def test_returns_conv_node_instance(self):
        model, node = make_conv_model(K=4, C=3, group=1)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert isinstance(result, cnodes.conv_node)

    def test_conv_node_has_matrix(self):
        K, C, kH, kW = 4, 3, 3, 3
        model, node = make_conv_model(K=K, C=C, kH=kH, kW=kW, group=1)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert hasattr(result, "matrix")
        # channel-major reshape: matrix is (C*kH*kW, K)
        assert result.matrix.shape == (C * kH * kW, K)

    def test_not_depthwise(self):
        model, node = make_conv_model(K=4, C=3, group=1)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert result.depthwise is False

    def test_result_is_not_a_list(self):
        model, node = make_conv_model(K=4, C=3, group=1)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: depthwise Conv (the bug-fix)
# ---------------------------------------------------------------------------

class TestDepthwiseConv:
    """Depthwise Conv (group > 1) must return convs + [catter] — both halves.

    Before the fix, ``return(convs)`` was followed by an unreachable
    ``return(catter)``, so the concatenator node was silently dropped from
    every depthwise-conv graph.
    """

    @pytest.fixture
    def K(self):
        return 4

    @pytest.fixture
    def dw_result(self, K):
        # Depthwise: group=K, C=K (one in-channel per group)
        model, node = make_conv_model(K=K, C=K, group=K)
        return cnode_factory.get_cnode_from_onnx_node(node, model)

    def test_returns_list(self, dw_result):
        assert isinstance(dw_result, list)

    def test_list_length_is_K_plus_one(self, dw_result, K):
        # K per-channel conv_nodes + 1 cat_node
        assert len(dw_result) == K + 1

    def test_last_element_is_cat_node(self, dw_result):
        """The concatenator must be present — it was the bug."""
        assert isinstance(dw_result[-1], cnodes.cat_node)

    def test_first_K_elements_are_conv_nodes(self, dw_result, K):
        for node in dw_result[:-1]:
            assert isinstance(node, cnodes.conv_node)

    def test_depthwise_conv_nodes_are_marked_depthwise(self, dw_result):
        for node in dw_result[:-1]:
            assert node.depthwise is True

    def test_cat_node_inputs_match_conv_outputs(self, dw_result):
        """cat_node.inputs must equal the outputs of every per-channel conv."""
        conv_outputs = [n.outputs[0] for n in dw_result[:-1]]
        cat_inputs = dw_result[-1].inputs
        assert conv_outputs == cat_inputs

    def test_two_channel_depthwise(self):
        """Regression: 2-channel depthwise also returns 2 convs + cat."""
        # K=2, C=2, group=2 → each group has 1 in-channel
        model, node = make_conv_model(K=2, C=2, group=2)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert len(result) == 3
        assert isinstance(result[-1], cnodes.cat_node)


# ---------------------------------------------------------------------------
# Tests: Gemm
# ---------------------------------------------------------------------------

class TestGemm:
    """Gemm node should return a gemm_node with a transposed matrix."""

    def test_returns_gemm_node(self):
        model, node = make_gemm_model(in_features=10, out_features=5)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert isinstance(result, cnodes.gemm_node)

    def test_gemm_matrix_shape_transposed(self):
        """from_onnx_node transposes B, so matrix should be (in, out)."""
        model, node = make_gemm_model(in_features=10, out_features=5)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert result.matrix.shape == (10, 5)

    def test_gemm_has_biases(self):
        model, node = make_gemm_model(in_features=10, out_features=5)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert hasattr(result, "biases")
        assert result.biases.shape == (5,)


# ---------------------------------------------------------------------------
# Tests: simple (no-initializer) node types
# ---------------------------------------------------------------------------

class TestSimpleNodes:
    """Smoke tests for node types that need no weight initializers."""

    def test_add_node(self):
        model, node = make_simple_node("Add", ["a", "b"], ["y"])
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert isinstance(result, cnodes.add_node)

    def test_global_average_pool(self):
        model, node = make_simple_node("GlobalAveragePool", ["x"], ["y"])
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert isinstance(result, cnodes.global_avg_node)

    def test_flatten(self):
        model, node = make_simple_node("Flatten", ["x"], ["y"])
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert isinstance(result, cnodes.flatten_node)

    def test_concat(self):
        model, node = make_simple_node("Concat", ["a", "b"], ["y"], axis=0)
        result = cnode_factory.get_cnode_from_onnx_node(node, model)
        assert isinstance(result, cnodes.concat_node)

    def test_unknown_op_raises_not_implemented(self):
        model, node = make_simple_node("Sigmoid", ["x"], ["y"])
        with pytest.raises(NotImplementedError, match="Sigmoid"):
            cnode_factory.get_cnode_from_onnx_node(node, model)


# ---------------------------------------------------------------------------
# Tests: Cgraph.get_matrix_dict()
# ---------------------------------------------------------------------------

class TestCgraphGetMatrixDict:
    """get_matrix_dict() must return a {int_index: np.ndarray} for every
    matrix-containing node in the cgraph."""

    @pytest.fixture
    def gemm_cgraph(self):
        """Build a minimal Cgraph from a single-Gemm ONNX model."""
        model, _ = make_gemm_model(in_features=8, out_features=4)
        return Cgraph.from_onnx_model(model)

    @pytest.fixture
    def conv_cgraph(self):
        """Build a minimal Cgraph from a single standard-Conv ONNX model."""
        model, _ = make_conv_model(K=4, C=3, group=1)
        return Cgraph.from_onnx_model(model)

    def test_returns_dict(self, gemm_cgraph):
        mx = gemm_cgraph.get_matrix_dict()
        assert isinstance(mx, dict)

    def test_gemm_matrix_present(self, gemm_cgraph):
        """A Gemm-only graph must have at least one entry (the Gemm's matrix)."""
        mx = gemm_cgraph.get_matrix_dict()
        assert len(mx) >= 1

    def test_conv_matrix_present(self, conv_cgraph):
        """A Conv-only graph must have at least one entry."""
        mx = conv_cgraph.get_matrix_dict()
        assert len(mx) >= 1

    def test_values_are_numpy_arrays(self, gemm_cgraph):
        mx = gemm_cgraph.get_matrix_dict()
        for v in mx.values():
            assert isinstance(v, np.ndarray)

    def test_keys_are_integer_indices(self, gemm_cgraph):
        mx = gemm_cgraph.get_matrix_dict()
        for k in mx.keys():
            assert isinstance(k, int)

    def test_gemm_matrix_shape(self, gemm_cgraph):
        """Verify the matrix shape matches what from_onnx_node produces."""
        mx = gemm_cgraph.get_matrix_dict()
        # in_features=8, out_features=4 → matrix is (8, 4) after transpose
        shapes = [v.shape for v in mx.values()]
        assert (8, 4) in shapes

    def test_conv_matrix_shape(self, conv_cgraph):
        """Verify the matrix shape matches: (C*kH*kW, K) = (3*3*3, 4)."""
        mx = conv_cgraph.get_matrix_dict()
        shapes = [v.shape for v in mx.values()]
        assert (27, 4) in shapes

    def test_no_matrix_nodes_excluded(self):
        """Nodes without a `matrix` attribute (e.g. add_node) must not appear."""
        model, _ = make_simple_node("Add", ["a", "b"], ["y"])
        # Build the cgraph manually — from_onnx_model would fail on a bare Add
        # because it has no initializers; construct directly instead.
        add = cnodes.add_node(["a", "b"], ["y"])
        cg = Cgraph([add])
        mx = cg.get_matrix_dict()
        assert len(mx) == 0

    def test_mixed_graph(self):
        """Only matrix-bearing nodes appear in get_matrix_dict."""
        gemm = cnodes.gemm_node(
            ["x"], ["y"],
            matrix=np.eye(4, dtype=np.float32),
            biases=np.zeros(4, dtype=np.float32),
        )
        add = cnodes.add_node(["y", "z"], ["out"])
        cg = Cgraph([gemm, add])
        mx = cg.get_matrix_dict()
        # Only the gemm_node has a matrix
        assert len(mx) == 1
        assert 0 in mx  # gemm is node index 0
