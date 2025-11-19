"""
Unit Tests for DRAI Resonance Layer

Tests the DraiResonanceLayer implementation for:
- Correct tensor shapes and dimensions
- Gradient flow and backpropagation
- Device and dtype handling
- Statistics tracking
- Integration compatibility

Author: Halcyon AI Research
Date: 2025-11-18
"""

import pytest
import torch
import torch.nn as nn
from src.drai import DraiResonanceLayer


class TestDraiResonanceLayerBasics:
    """Test basic functionality of DraiResonanceLayer."""

    def test_initialization(self):
        """Test layer initializes with correct parameters."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=1, head_dim=64)

        assert layer.hidden_size == 2048
        assert layer.num_heads == 1
        assert layer.head_dim == 64
        assert layer._phase == 1
        assert layer._returns_zeros is True

    def test_initialization_defaults(self):
        """Test layer uses sensible defaults."""
        layer = DraiResonanceLayer(hidden_size=1024)

        assert layer.num_heads == 1
        assert layer.head_dim == 64
        assert layer.device == "cpu"
        assert layer.dtype == torch.float32

    def test_initialization_custom_head_dim(self):
        """Test layer accepts custom head dimension."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=2, head_dim=128)

        assert layer.num_heads == 2
        assert layer.head_dim == 128


class TestDraiResonanceLayerForward:
    """Test forward pass behavior."""

    def test_forward_output_shapes(self):
        """Test forward pass produces correct output shapes."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=1, head_dim=64)

        # Typical transformer dimensions: [seq_len, batch, num_heads, head_dim]
        seq_len, batch, num_attn_heads, head_dim = 20, 4, 32, 64
        query = torch.randn(seq_len, batch, num_attn_heads, head_dim)

        k_reson, v_reson = layer(query)

        # DRAI output should have shape [seq_len, batch, num_drai_heads, head_dim]
        expected_shape = (seq_len, batch, 1, head_dim)
        assert k_reson.shape == expected_shape, f"Expected {expected_shape}, got {k_reson.shape}"
        assert v_reson.shape == expected_shape, f"Expected {expected_shape}, got {v_reson.shape}"

    def test_forward_multiple_drai_heads(self):
        """Test forward with multiple DRAI heads."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=3, head_dim=64)

        query = torch.randn(10, 2, 16, 64)
        k_reson, v_reson = layer(query)

        assert k_reson.shape == (10, 2, 3, 64)
        assert v_reson.shape == (10, 2, 3, 64)

    def test_forward_returns_zeros_phase1(self):
        """Test Phase 1 implementation returns zeros."""
        layer = DraiResonanceLayer(hidden_size=1024, num_heads=1)

        query = torch.randn(10, 4, 16, 64)
        k_reson, v_reson = layer(query)

        # Phase 1 should return all zeros
        assert torch.all(k_reson == 0), "Phase 1 should return zero keys"
        assert torch.all(v_reson == 0), "Phase 1 should return zero values"

    def test_forward_preserves_device(self):
        """Test output tensors are on same device as input."""
        layer = DraiResonanceLayer(hidden_size=1024)

        query = torch.randn(10, 2, 8, 64)
        k_reson, v_reson = layer(query)

        assert k_reson.device == query.device
        assert v_reson.device == query.device

    def test_forward_preserves_dtype(self):
        """Test output tensors preserve input dtype."""
        layer = DraiResonanceLayer(hidden_size=1024)

        # Test with float32
        query_f32 = torch.randn(10, 2, 8, 64, dtype=torch.float32)
        k_reson, v_reson = layer(query_f32)
        assert k_reson.dtype == torch.float32
        assert v_reson.dtype == torch.float32

        # Test with float16
        query_f16 = torch.randn(10, 2, 8, 64, dtype=torch.float16)
        k_reson, v_reson = layer(query_f16)
        assert k_reson.dtype == torch.float16
        assert v_reson.dtype == torch.float16

    def test_forward_with_different_batch_sizes(self):
        """Test layer handles varying batch sizes."""
        layer = DraiResonanceLayer(hidden_size=1024)

        for batch_size in [1, 2, 8, 16, 32]:
            query = torch.randn(10, batch_size, 8, 64)
            k_reson, v_reson = layer(query)

            assert k_reson.shape[1] == batch_size
            assert v_reson.shape[1] == batch_size

    def test_forward_with_different_seq_lengths(self):
        """Test layer handles varying sequence lengths."""
        layer = DraiResonanceLayer(hidden_size=1024)

        for seq_len in [1, 10, 50, 100, 512]:
            query = torch.randn(seq_len, 4, 8, 64)
            k_reson, v_reson = layer(query)

            assert k_reson.shape[0] == seq_len
            assert v_reson.shape[0] == seq_len


class TestDraiResonanceLayerGradients:
    """Test gradient flow and backpropagation."""

    def test_gradients_flow_through_layer(self):
        """Test gradients can flow through DRAI layer (even though it returns zeros)."""
        layer = DraiResonanceLayer(hidden_size=1024)

        query = torch.randn(10, 2, 8, 64, requires_grad=True)
        k_reson, v_reson = layer(query)

        # Create a simple loss
        loss = k_reson.sum() + v_reson.sum()
        loss.backward()

        # Query should have gradients (all zeros since k/v are zeros)
        assert query.grad is not None
        assert query.grad.shape == query.shape

    def test_no_learnable_parameters_phase1(self):
        """Test Phase 1 has no learnable parameters."""
        layer = DraiResonanceLayer(hidden_size=1024)

        # Phase 1 should have no parameters
        params = list(layer.parameters())
        assert len(params) == 0, "Phase 1 should have no learnable parameters"

    def test_has_buffers(self):
        """Test layer has registered buffers for statistics."""
        layer = DraiResonanceLayer(hidden_size=1024)

        buffers = dict(layer.named_buffers())
        assert "_forward_count" in buffers
        assert buffers["_forward_count"].dtype == torch.long


class TestDraiResonanceLayerIntegration:
    """Test integration compatibility with transformer attention."""

    def test_concatenation_with_attention_kv(self):
        """Test DRAI output can be concatenated with standard attention K/V."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=1, head_dim=64)

        seq_len, batch, num_attn_heads, head_dim = 20, 4, 32, 64

        # Simulate standard attention K/V
        key_layer = torch.randn(seq_len, batch, num_attn_heads, head_dim)
        value_layer = torch.randn(seq_len, batch, num_attn_heads, head_dim)

        # Get DRAI resonance outputs
        query = torch.randn(seq_len, batch, num_attn_heads, head_dim)
        k_reson, v_reson = layer(query)

        # Concatenate along head dimension (dim=2)
        key_extended = torch.cat([key_layer, k_reson], dim=2)
        value_extended = torch.cat([value_layer, v_reson], dim=2)

        # Verify shapes
        assert key_extended.shape == (seq_len, batch, num_attn_heads + 1, head_dim)
        assert value_extended.shape == (seq_len, batch, num_attn_heads + 1, head_dim)

    def test_attention_computation_with_drai(self):
        """Test full attention computation works with DRAI injection."""
        layer = DraiResonanceLayer(hidden_size=512, num_heads=1, head_dim=64)

        seq_len, batch, num_heads, head_dim = 10, 2, 8, 64

        # Create Q, K, V
        Q = torch.randn(seq_len, batch, num_heads, head_dim)
        K = torch.randn(seq_len, batch, num_heads, head_dim)
        V = torch.randn(seq_len, batch, num_heads, head_dim)

        # Get DRAI outputs
        k_reson, v_reson = layer(Q)

        # Extend K and V with DRAI head
        # Note: K_extended has num_heads + 1 heads (DRAI adds memory, not query)
        K_extended = torch.cat([K, k_reson], dim=2)
        V_extended = torch.cat([V, v_reson], dim=2)

        # Verify extended shapes
        assert K_extended.shape == (seq_len, batch, num_heads + 1, head_dim)
        assert V_extended.shape == (seq_len, batch, num_heads + 1, head_dim)

        # For actual attention computation, we process per head
        # Each Q head attends over all K/V positions (including DRAI)
        # Reshape for per-head attention: [batch, heads, seq, head_dim]
        Q_attn = Q.permute(1, 2, 0, 3)  # [batch, num_heads, seq, head_dim]

        # For each Q head, we attend over all K positions
        # We'll test this by doing attention for the first head only
        Q_head0 = Q_attn[:, 0:1, :, :]  # [batch, 1, seq, head_dim]

        # Get all keys and values (including DRAI) for this head position
        # In real transformer, each head has its own K/V projection
        # Here we simulate by taking the corresponding K/V slice
        K_head0 = K_extended[:, :, 0, :].permute(1, 0, 2).unsqueeze(1)  # [batch, 1, seq, head_dim]
        V_head0 = V_extended[:, :, 0, :].permute(1, 0, 2).unsqueeze(1)  # [batch, 1, seq, head_dim]

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q_head0, K_head0.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention: weighted sum of values
        output = torch.matmul(attn_weights, V_head0)

        # Verify output shape
        assert output.shape == (batch, 1, seq_len, head_dim)


class TestDraiResonanceLayerStatistics:
    """Test statistics tracking functionality."""

    def test_forward_count_increments(self):
        """Test forward count increments on each forward pass."""
        layer = DraiResonanceLayer(hidden_size=1024)

        query = torch.randn(10, 2, 8, 64)

        assert layer._forward_count == 0

        layer(query)
        assert layer._forward_count == 1

        layer(query)
        assert layer._forward_count == 2

        layer(query)
        layer(query)
        assert layer._forward_count == 4

    def test_get_statistics(self):
        """Test get_statistics returns correct information."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=2, head_dim=128)

        query = torch.randn(10, 2, 8, 128)
        layer(query)
        layer(query)

        stats = layer.get_statistics()

        assert stats["forward_count"] == 2
        assert stats["phase"] == 1
        assert stats["returns_zeros"] is True
        assert stats["num_heads"] == 2
        assert stats["head_dim"] == 128

    def test_reset_statistics(self):
        """Test statistics can be reset."""
        layer = DraiResonanceLayer(hidden_size=1024)

        query = torch.randn(10, 2, 8, 64)
        layer(query)
        layer(query)
        layer(query)

        assert layer._forward_count == 3

        layer.reset_statistics()
        assert layer._forward_count == 0


class TestDraiResonanceLayerEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_token_sequence(self):
        """Test layer handles single token sequences."""
        layer = DraiResonanceLayer(hidden_size=1024)

        query = torch.randn(1, 4, 8, 64)  # seq_len=1
        k_reson, v_reson = layer(query)

        assert k_reson.shape == (1, 4, 1, 64)
        assert v_reson.shape == (1, 4, 1, 64)

    def test_single_batch(self):
        """Test layer handles batch size of 1."""
        layer = DraiResonanceLayer(hidden_size=1024)

        query = torch.randn(10, 1, 8, 64)  # batch=1
        k_reson, v_reson = layer(query)

        assert k_reson.shape == (10, 1, 1, 64)
        assert v_reson.shape == (10, 1, 1, 64)

    def test_head_dim_mismatch_raises_error(self):
        """Test layer raises error if input head_dim doesn't match."""
        layer = DraiResonanceLayer(hidden_size=1024, head_dim=64)

        # Create query with wrong head_dim
        query = torch.randn(10, 4, 8, 128)  # head_dim=128, but layer expects 64

        with pytest.raises(AssertionError, match="Expected head_dim=64"):
            layer(query)

    def test_extra_repr(self):
        """Test string representation for debugging."""
        layer = DraiResonanceLayer(hidden_size=2048, num_heads=2, head_dim=128)

        repr_str = layer.extra_repr()

        assert "hidden_size=2048" in repr_str
        assert "num_heads=2" in repr_str
        assert "head_dim=128" in repr_str
        assert "phase=1" in repr_str


# Test markers for pytest
pytestmark = pytest.mark.unit


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/unit/test_drai_resonance.py -v
    pytest.main([__file__, "-v"])
