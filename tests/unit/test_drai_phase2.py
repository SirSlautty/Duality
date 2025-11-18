"""
Unit Tests for DRAI Phase 2 - Attractor Dynamics

Tests the Phase 2 implementation of DraiResonanceLayer including:
- Attractor formation from patterns
- Pattern reinforcement via EMA
- Cosine similarity matching
- Exponential decay dynamics
- K/V generation from attractors
- Statistics and metadata tracking

Author: Halcyon AI Research
Date: 2025-11-18
"""

import pytest
import torch
import torch.nn.functional as F
from src.drai import DraiResonanceLayer


class TestDraiPhase2Initialization:
    """Test Phase 2 initialization and configuration."""

    def test_phase2_initialization(self):
        """Test Phase 2 layer initializes with attractor buffers."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2)

        assert layer._phase == 2
        assert layer._returns_zeros is False
        assert hasattr(layer, 'attractor_centroids')
        assert hasattr(layer, 'attractor_coherence')
        assert hasattr(layer, 'attractor_last_used')
        assert hasattr(layer, 'attractor_reinforcements')
        assert hasattr(layer, 'attractor_count')
        assert hasattr(layer, 'timestep')

    def test_phase2_buffer_shapes(self):
        """Test attractor buffers have correct shapes."""
        max_attractors = 16
        head_dim = 64
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            max_attractors=max_attractors,
            head_dim=head_dim
        )

        assert layer.attractor_centroids.shape == (max_attractors, head_dim)
        assert layer.attractor_coherence.shape == (max_attractors,)
        assert layer.attractor_last_used.shape == (max_attractors,)
        assert layer.attractor_reinforcements.shape == (max_attractors,)
        assert layer.attractor_count.shape == ()
        assert layer.timestep.shape == ()

    def test_phase2_initial_state(self):
        """Test attractor field starts empty."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2)

        assert layer.attractor_count.item() == 0
        assert layer.timestep.item() == 0
        assert torch.all(layer.attractor_centroids == 0)
        assert torch.all(layer.attractor_coherence == 0)

    def test_phase2_custom_parameters(self):
        """Test custom Phase 2 hyperparameters."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            max_attractors=64,
            coherence_threshold=0.5,
            formation_threshold=0.7,
            decay_rate=0.05,
            ema_momentum=0.95,
        )

        assert layer.max_attractors == 64
        assert layer.coherence_threshold == 0.5
        assert layer.formation_threshold == 0.7
        assert layer.decay_rate == 0.05
        assert layer.ema_momentum == 0.95


class TestDraiPhase2AttractorFormation:
    """Test attractor creation and formation dynamics."""

    def test_first_pattern_creates_attractor(self):
        """Test first forward pass creates an attractor."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)
        k_reson, v_reson = layer(query)

        # Should have created one attractor
        assert layer.attractor_count.item() == 1
        assert layer.timestep.item() == 1

        # Attractor should have been initialized
        assert torch.any(layer.attractor_centroids[0] != 0)
        assert layer.attractor_coherence[0].item() == layer.formation_threshold
        assert layer.attractor_last_used[0].item() == 0
        assert layer.attractor_reinforcements[0].item() == 1

    def test_multiple_patterns_create_multiple_attractors(self):
        """Test different patterns create distinct attractors."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            coherence_threshold=0.9  # High threshold to force new attractors
        )

        # Create 5 distinct random patterns
        for i in range(5):
            query = torch.randn(10, 2, 8, 64)
            layer(query)

        # Should have created 5 attractors
        assert layer.attractor_count.item() == 5

        # All should have coherence close to initial (may have decayed slightly)
        for i in range(5):
            coherence = layer.attractor_coherence[i].item()
            assert 0.4 <= coherence <= 0.6  # Around formation_threshold (0.5)
            assert layer.attractor_reinforcements[i].item() == 1

    def test_similar_patterns_reinforce_attractor(self):
        """Test similar patterns reinforce the same attractor."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            coherence_threshold=0.3  # Low threshold for easier matching
        )

        # Create base pattern
        base_pattern = torch.randn(10, 2, 8, 64)
        layer(base_pattern)

        initial_coherence = layer.attractor_coherence[0].item()
        initial_centroid = layer.attractor_centroids[0].clone()

        # Create very similar pattern (base + small noise)
        similar_pattern = base_pattern + torch.randn_like(base_pattern) * 0.01
        layer(similar_pattern)

        # Should still have only 1 attractor (reinforced, not created new)
        assert layer.attractor_count.item() == 1

        # Coherence should have increased
        assert layer.attractor_coherence[0].item() > initial_coherence

        # Centroid should have moved slightly (EMA update)
        assert not torch.allclose(layer.attractor_centroids[0], initial_centroid)

        # Reinforcement count should increase
        assert layer.attractor_reinforcements[0].item() == 2

    def test_capacity_management(self):
        """Test attractor replacement when capacity is reached."""
        max_attractors = 4
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            max_attractors=max_attractors,
            coherence_threshold=0.9,  # High to force new attractors
        )

        # Fill capacity
        for i in range(max_attractors):
            query = torch.randn(10, 2, 8, 64)
            layer(query)

        assert layer.attractor_count.item() == max_attractors

        # Add one more pattern - should replace weakest
        query = torch.randn(10, 2, 8, 64)
        layer(query)

        # Count should stay at max
        assert layer.attractor_count.item() == max_attractors

        # One attractor should have been replaced (new one has reinforcement=1)
        reinforcement_counts = [layer.attractor_reinforcements[i].item()
                               for i in range(max_attractors)]
        assert 1 in reinforcement_counts


class TestDraiPhase2PatternExtraction:
    """Test pattern extraction from query layers."""

    def test_pattern_extraction_shape(self):
        """Test extracted pattern has correct shape."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)
        pattern = layer._extract_pattern(query)

        # Should be [head_dim]
        assert pattern.shape == (64,)

    def test_pattern_is_normalized(self):
        """Test extracted pattern is normalized to unit sphere."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)
        pattern = layer._extract_pattern(query)

        # Should have L2 norm ≈ 1
        norm = torch.norm(pattern, p=2).item()
        assert abs(norm - 1.0) < 1e-5

    def test_pattern_extraction_deterministic(self):
        """Test same input produces same pattern."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)
        pattern1 = layer._extract_pattern(query)
        pattern2 = layer._extract_pattern(query)

        assert torch.allclose(pattern1, pattern2)


class TestDraiPhase2CosineSimilarity:
    """Test cosine similarity-based pattern matching."""

    def test_find_best_match_no_attractors(self):
        """Test matching when no attractors exist."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        pattern = F.normalize(torch.randn(64), p=2, dim=-1)
        best_idx, best_sim, _ = layer._find_best_match(pattern)

        assert best_idx == -1
        assert best_sim == -1.0

    def test_find_best_match_single_attractor(self):
        """Test matching with one attractor."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        # Create one attractor
        query = torch.randn(10, 2, 8, 64)
        layer(query)

        # Extract the pattern and match
        pattern = layer._extract_pattern(query)
        best_idx, best_sim, _ = layer._find_best_match(pattern)

        assert best_idx == 0
        # Should be very similar (almost 1.0) since it's the same pattern
        assert best_sim > 0.9

    def test_find_best_match_multiple_attractors(self):
        """Test matching selects most similar attractor."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            coherence_threshold=0.9  # Force creation of multiple attractors
        )

        # Create 3 distinct attractors
        patterns = []
        for i in range(3):
            query = torch.randn(10, 2, 8, 64)
            layer(query)
            patterns.append(layer._extract_pattern(query))

        # Match against second pattern
        best_idx, best_sim, _ = layer._find_best_match(patterns[1])

        assert best_idx == 1  # Should match second attractor
        assert best_sim > 0.9  # High similarity


class TestDraiPhase2Reinforcement:
    """Test EMA-based attractor reinforcement."""

    def test_reinforcement_updates_centroid(self):
        """Test reinforcement updates attractor centroid via EMA."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64, ema_momentum=0.9)

        # Create initial attractor
        query1 = torch.randn(10, 2, 8, 64)
        layer(query1)

        initial_centroid = layer.attractor_centroids[0].clone()

        # Reinforce with similar pattern
        query2 = query1 + torch.randn_like(query1) * 0.1
        layer(query2)

        new_centroid = layer.attractor_centroids[0]

        # Centroid should have changed
        assert not torch.allclose(initial_centroid, new_centroid)

        # But not too much (due to EMA with momentum 0.9)
        change = torch.norm(new_centroid - initial_centroid).item()
        assert change < 0.5  # Should be small change

    def test_reinforcement_increases_coherence(self):
        """Test reinforcement increases attractor coherence."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)
        layer(query)

        initial_coherence = layer.attractor_coherence[0].item()

        # Reinforce multiple times
        for _ in range(5):
            layer(query)

        final_coherence = layer.attractor_coherence[0].item()

        # Coherence should increase
        assert final_coherence > initial_coherence

        # But stay bounded by 1.0
        assert final_coherence <= 1.0

    def test_reinforcement_maintains_normalization(self):
        """Test attractor remains normalized after reinforcement."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)

        # Reinforce many times
        for _ in range(10):
            layer(query + torch.randn_like(query) * 0.1)

        # Centroids should still be normalized
        for i in range(layer.attractor_count.item()):
            norm = torch.norm(layer.attractor_centroids[i], p=2).item()
            assert abs(norm - 1.0) < 1e-5


class TestDraiPhase2Decay:
    """Test exponential decay dynamics."""

    def test_decay_reduces_coherence(self):
        """Test unused attractors decay over time."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            decay_rate=0.1,  # Faster decay for testing
            coherence_threshold=0.2,  # Low to ensure attractors can be reinforced
        )

        # Create two very different attractors using orthogonal patterns
        # This ensures they won't accidentally match each other
        query1 = torch.zeros(10, 2, 8, 64)
        query1[:, :, :, :32] = 1.0  # First half = 1

        query2 = torch.zeros(10, 2, 8, 64)
        query2[:, :, :, 32:] = 1.0  # Second half = 1

        layer(query1)
        layer(query2)

        coherence_0_initial = layer.attractor_coherence[0].item()
        coherence_1_initial = layer.attractor_coherence[1].item()

        # Keep using only the second attractor
        for _ in range(5):
            layer(query2)

        coherence_0_final = layer.attractor_coherence[0].item()
        coherence_1_final = layer.attractor_coherence[1].item()

        # First attractor should have decayed (not reinforced)
        assert coherence_0_final < coherence_0_initial

        # Second attractor should have increased (reinforced)
        assert coherence_1_final > coherence_1_initial

    def test_decay_prunes_weak_attractors(self):
        """Test very weak attractors get pruned."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            decay_rate=0.5,  # Very fast decay
            coherence_threshold=0.9,
            formation_threshold=0.1,  # Low initial coherence
        )

        # Create attractor
        query1 = torch.randn(10, 2, 8, 64)
        layer(query1)

        assert layer.attractor_count.item() == 1

        # Use different patterns to decay the first
        for _ in range(20):
            query_new = torch.randn(10, 2, 8, 64)
            layer(query_new)

        # First attractor should have been pruned
        # (exact count depends on coherence_threshold and how many new ones formed)
        # But the decayed one should be gone
        assert layer._attractors_decayed.item() > 0


class TestDraiPhase2KVGeneration:
    """Test synthetic K/V generation from attractors."""

    def test_kv_generation_with_no_attractors(self):
        """Test K/V generation returns zeros when no attractors exist."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64, num_heads=1)

        # Don't create any attractors, just call _generate_kv directly
        k_reson, v_reson = layer._generate_kv(10, 2, torch.device('cpu'), torch.float32)

        assert torch.all(k_reson == 0)
        assert torch.all(v_reson == 0)
        assert k_reson.shape == (10, 2, 1, 64)
        assert v_reson.shape == (10, 2, 1, 64)

    def test_kv_generation_with_attractors(self):
        """Test K/V generation produces non-zero values with attractors."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64, num_heads=1)

        # Create attractors
        query = torch.randn(10, 2, 8, 64)
        k_reson, v_reson = layer(query)

        # Should be non-zero (attractor exists)
        assert torch.any(k_reson != 0)
        assert torch.any(v_reson != 0)

    def test_kv_values_based_on_strongest_attractors(self):
        """Test K/V uses strongest attractors."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            num_heads=1,
            coherence_threshold=0.9,  # Force multiple attractors
        )

        # Create multiple attractors with different strengths
        queries = [torch.randn(10, 2, 8, 64) for _ in range(3)]
        for q in queries:
            layer(q)

        # Reinforce first attractor more
        for _ in range(10):
            layer(queries[0])

        # First attractor should be strongest
        assert layer.attractor_coherence[0] > layer.attractor_coherence[1]
        assert layer.attractor_coherence[0] > layer.attractor_coherence[2]

        # K/V should be based on first attractor
        k_reson, v_reson = layer._generate_kv(10, 2, torch.device('cpu'), torch.float32)

        # Key should match strongest attractor centroid
        k_first_head = k_reson[0, 0, 0, :]
        strongest_centroid = layer.attractor_centroids[0]

        assert torch.allclose(k_first_head, strongest_centroid, atol=1e-6)

    def test_v_weighted_by_coherence(self):
        """Test V values are weighted by coherence."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64, num_heads=1)

        query = torch.randn(10, 2, 8, 64)
        layer(query)

        k_reson, v_reson = layer._generate_kv(10, 2, torch.device('cpu'), torch.float32)

        # V should be coherence * centroid
        expected_v = layer.attractor_coherence[0] * layer.attractor_centroids[0]
        actual_v = v_reson[0, 0, 0, :]

        assert torch.allclose(actual_v, expected_v, atol=1e-6)


class TestDraiPhase2Statistics:
    """Test Phase 2 statistics tracking."""

    def test_statistics_include_phase2_metrics(self):
        """Test get_statistics returns Phase 2 metrics."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2)

        query = torch.randn(10, 2, 8, 64)
        layer(query)

        stats = layer.get_statistics()

        # Should have Phase 2 metrics
        assert 'attractor_count' in stats
        assert 'attractors_created' in stats
        assert 'attractors_reinforced' in stats
        assert 'attractors_decayed' in stats
        assert 'timestep' in stats
        assert 'max_attractors' in stats
        assert 'utilization' in stats

    def test_statistics_track_creation(self):
        """Test creation statistics are tracked."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, coherence_threshold=0.9)

        # Create 3 attractors
        for _ in range(3):
            layer(torch.randn(10, 2, 8, 64))

        stats = layer.get_statistics()
        assert stats['attractors_created'] == 3
        assert stats['attractor_count'] == 3

    def test_statistics_track_reinforcement(self):
        """Test reinforcement statistics are tracked."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2)

        query = torch.randn(10, 2, 8, 64)

        # First creates, subsequent reinforces
        layer(query)
        layer(query)
        layer(query)

        stats = layer.get_statistics()
        assert stats['attractors_created'] == 1
        assert stats['attractors_reinforced'] >= 2  # At least 2 reinforcements

    def test_get_attractor_info(self):
        """Test get_attractor_info returns detailed information."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64)
        layer(query)

        info = layer.get_attractor_info()

        assert 'centroids' in info
        assert 'coherence' in info
        assert 'last_used' in info
        assert 'reinforcements' in info

        # Check shapes
        assert info['centroids'].shape == (1, 64)  # 1 attractor, 64 dims
        assert info['coherence'].shape == (1,)
        assert info['last_used'].shape == (1,)
        assert info['reinforcements'].shape == (1,)


class TestDraiPhase2Integration:
    """Test Phase 2 integration scenarios."""

    def test_full_forward_pass_phase2(self):
        """Test complete forward pass with Phase 2."""
        layer = DraiResonanceLayer(hidden_size=2048, phase=2, head_dim=64, num_heads=1)

        seq_len, batch, num_heads, head_dim = 20, 4, 32, 64
        query = torch.randn(seq_len, batch, num_heads, head_dim)

        k_reson, v_reson = layer(query)

        # Correct shapes
        assert k_reson.shape == (seq_len, batch, 1, head_dim)
        assert v_reson.shape == (seq_len, batch, 1, head_dim)

        # Should have created attractor
        assert layer.attractor_count.item() >= 1

        # Should have non-zero values
        assert torch.any(k_reson != 0)
        assert torch.any(v_reson != 0)

    def test_repeated_sequence_reinforcement(self):
        """Test repeated sequences reinforce the same attractor."""
        layer = DraiResonanceLayer(
            hidden_size=1024,
            phase=2,
            head_dim=64,
            coherence_threshold=0.3,
        )

        # Create repeated pattern
        pattern = torch.randn(10, 2, 8, 64)

        # Process 10 times
        for _ in range(10):
            layer(pattern)

        # Should have created only 1 attractor (or very few)
        assert layer.attractor_count.item() <= 2

        # First attractor should be strongly reinforced
        assert layer.attractor_reinforcements[0].item() >= 5

        # High coherence
        assert layer.attractor_coherence[0].item() > 0.7

    def test_gradients_flow_through_phase2(self):
        """Test gradients can flow through Phase 2."""
        layer = DraiResonanceLayer(hidden_size=1024, phase=2, head_dim=64)

        query = torch.randn(10, 2, 8, 64, requires_grad=True)
        k_reson, v_reson = layer(query)

        # Create loss
        loss = k_reson.sum() + v_reson.sum()
        loss.backward()

        # Gradients should exist
        assert query.grad is not None
        assert query.grad.shape == query.shape


# Test markers
pytestmark = pytest.mark.unit


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/unit/test_drai_phase2.py -v
    pytest.main([__file__, "-v"])
