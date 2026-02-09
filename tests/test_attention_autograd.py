"""Tests for select, stack ops and 3D batched attention autograd connectivity."""

import pytest
from quantml.tensor import Tensor
from quantml import ops
from quantml.autograd import build_topo


class TestSelect:
    def test_forward(self):
        data_3d = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        t = Tensor(data_3d, requires_grad=True)

        s0 = ops.select(t, 0, dim=0)
        assert s0.shape == (2, 2)
        assert s0.data[0][0] == pytest.approx(1.0)
        assert s0.data[1][1] == pytest.approx(4.0)

        s1 = ops.select(t, 1, dim=0)
        assert s1.shape == (2, 2)
        assert s1.data[0][0] == pytest.approx(5.0)
        assert s1.data[1][1] == pytest.approx(8.0)

    def test_backward(self):
        data_3d = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        t = Tensor(data_3d, requires_grad=True)

        s = ops.select(t, 0, dim=0)
        loss = ops.sum(s)
        loss.backward()

        grad = t.grad
        if hasattr(grad, 'tolist'):
            grad = grad.tolist()
        # Batch 0 should have gradient 1.0, batch 1 should have 0.0
        assert grad[0][0][0] == pytest.approx(1.0)
        assert grad[0][1][1] == pytest.approx(1.0)
        assert grad[1][0][0] == pytest.approx(0.0)
        assert grad[1][1][1] == pytest.approx(0.0)

    def test_single_batch(self):
        data_3d = [[[1.0, 2.0], [3.0, 4.0]]]
        t = Tensor(data_3d, requires_grad=True)

        s = ops.select(t, 0, dim=0)
        loss = ops.sum(s)
        loss.backward()

        grad = t.grad
        if hasattr(grad, 'tolist'):
            grad = grad.tolist()
        assert grad[0][0][0] == pytest.approx(1.0)
        assert grad[0][1][1] == pytest.approx(1.0)

    def test_graph_connection(self):
        data_3d = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        t = Tensor(data_3d, requires_grad=True)
        s = ops.select(t, 0, dim=0)

        topo = build_topo(s)
        assert t in topo


class TestStack:
    def test_forward(self):
        t0 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t1 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        stacked = ops.stack([t0, t1], dim=0)
        assert stacked.shape == (2, 2, 2)
        assert stacked.data[0][0][0] == pytest.approx(1.0)
        assert stacked.data[1][1][1] == pytest.approx(8.0)

    def test_backward(self):
        t0 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        t1 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        stacked = ops.stack([t0, t1], dim=0)
        loss = ops.sum(stacked)
        loss.backward()

        g0 = t0.grad
        g1 = t1.grad
        if hasattr(g0, 'tolist'):
            g0 = g0.tolist()
        if hasattr(g1, 'tolist'):
            g1 = g1.tolist()
        assert g0[0][0] == pytest.approx(1.0)
        assert g0[1][1] == pytest.approx(1.0)
        assert g1[0][0] == pytest.approx(1.0)
        assert g1[1][1] == pytest.approx(1.0)

    def test_graph_connection(self):
        t0 = Tensor([[1.0, 2.0]], requires_grad=True)
        t1 = Tensor([[3.0, 4.0]], requires_grad=True)

        stacked = ops.stack([t0, t1], dim=0)
        topo = build_topo(stacked)
        assert t0 in topo
        assert t1 in topo


class TestSelectStackRoundtrip:
    def test_preserves_graph(self):
        data_3d = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        t = Tensor(data_3d, requires_grad=True)

        slices = []
        for b in range(2):
            s = ops.select(t, b, dim=0)
            s_transformed = ops.mul(s, 2.0)
            slices.append(s_transformed)

        result = ops.stack(slices, dim=0)
        loss = ops.sum(result)
        loss.backward()

        grad = t.grad
        if hasattr(grad, 'tolist'):
            grad = grad.tolist()
        # d/dt of sum(2*t) = 2 everywhere
        for b in range(2):
            for r in range(2):
                for c in range(2):
                    assert grad[b][r][c] == pytest.approx(2.0, abs=0.1)


class TestSelfAttention3DAutograd:
    def test_graph_connected(self):
        from quantml.models.attention import SelfAttention

        embed_dim = 4
        attn = SelfAttention(embed_dim)

        x_data = [[[float(b * 12 + s * 4 + d) * 0.1 for d in range(4)]
                    for s in range(3)]
                   for b in range(2)]
        x = Tensor(x_data, requires_grad=True)

        out = attn.forward(x)

        topo = build_topo(out)
        assert x in topo, "Input tensor should be reachable in the computation graph"

    def test_input_gets_gradient(self):
        from quantml.models.attention import SelfAttention

        embed_dim = 4
        attn = SelfAttention(embed_dim)

        x_data = [[[float(b * 12 + s * 4 + d) * 0.1 for d in range(4)]
                    for s in range(3)]
                   for b in range(2)]
        x = Tensor(x_data, requires_grad=True)

        out = attn.forward(x)
        loss = ops.sum(out)
        loss.backward()

        assert x.grad is not None, "Input tensor should have a gradient"

    def test_param_gradients(self):
        from quantml.models.attention import SelfAttention

        embed_dim = 4
        attn = SelfAttention(embed_dim)

        x_data = [[[0.1 * (b * 12 + s * 4 + d) for d in range(4)]
                    for s in range(3)]
                   for b in range(2)]
        x = Tensor(x_data, requires_grad=True)

        out = attn.forward(x)
        loss = ops.sum(out)
        loss.backward()

        for p in attn.parameters():
            assert p.grad is not None, f"Parameter with shape {p.shape} has no gradient"

    def test_2d_still_works(self):
        """Regression test: 2D attention path is not broken."""
        from quantml.models.attention import SelfAttention

        embed_dim = 4
        attn = SelfAttention(embed_dim)

        x_data = [[0.1 * (s * 4 + d) for d in range(4)] for s in range(3)]
        x = Tensor(x_data, requires_grad=True)

        out = attn.forward(x)
        loss = ops.sum(out)
        loss.backward()

        assert x.grad is not None
        for p in attn.parameters():
            assert p.grad is not None


class TestMultiHeadAttention3DAutograd:
    def test_graph_connected(self):
        from quantml.models.attention import MultiHeadAttention

        embed_dim = 4
        num_heads = 2
        mha = MultiHeadAttention(embed_dim, num_heads)

        x_data = [[[float(b * 12 + s * 4 + d) * 0.1 for d in range(4)]
                    for s in range(3)]
                   for b in range(2)]
        x = Tensor(x_data, requires_grad=True)

        out = mha.forward(x)

        topo = build_topo(out)
        assert x in topo

    def test_input_gets_gradient(self):
        from quantml.models.attention import MultiHeadAttention

        embed_dim = 4
        num_heads = 2
        mha = MultiHeadAttention(embed_dim, num_heads)

        x_data = [[[float(b * 12 + s * 4 + d) * 0.1 for d in range(4)]
                    for s in range(3)]
                   for b in range(2)]
        x = Tensor(x_data, requires_grad=True)

        out = mha.forward(x)
        loss = ops.sum(out)
        loss.backward()

        assert x.grad is not None
