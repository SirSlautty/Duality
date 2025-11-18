#!/usr/bin/env python3
"""
PyTorch Installation Test Suite for Project Duality
Tests core PyTorch functionality needed for transformer and DRAI development.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_basic_installation():
    """Test basic PyTorch installation and configuration."""
    print_header("1. Basic Installation Info")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return True

def test_tensor_operations():
    """Test basic tensor operations."""
    print_header("2. Tensor Operations")

    # Create tensors
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    print(f"Created tensors x and y with shape: {x.shape}")

    # Basic operations
    z = x + y
    print(f"Addition: x + y shape: {z.shape}")

    # Matrix multiplication
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.matmul(a, b)
    print(f"Matrix multiplication: (3x4) @ (4x5) = {c.shape}")

    # Broadcasting
    d = torch.randn(3, 1)
    e = d + x
    print(f"Broadcasting: (3x1) + (3x4) = {e.shape}")

    print("✓ All tensor operations passed")
    return True

def test_neural_network_basics():
    """Test basic neural network components."""
    print_header("3. Neural Network Basics")

    # Linear layer
    linear = nn.Linear(128, 256)
    x = torch.randn(32, 128)  # batch_size=32, features=128
    y = linear(x)
    print(f"Linear layer: input {x.shape} -> output {y.shape}")

    # Activation functions
    relu_out = F.relu(y)
    gelu_out = F.gelu(y)
    print(f"Activations (ReLU, GELU): {relu_out.shape}, {gelu_out.shape}")

    # LayerNorm (commonly used in transformers)
    ln = nn.LayerNorm(256)
    norm_out = ln(y)
    print(f"LayerNorm: {norm_out.shape}")

    print("✓ All neural network basics passed")
    return True

def test_attention_mechanism():
    """Test attention mechanism - critical for Duality project."""
    print_header("4. Attention Mechanism")

    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 4
    head_dim = d_model // num_heads

    # Create Q, K, V tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"Q, K, V shapes: {Q.shape}")

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"Attention scores shape: {scores.shape}")

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    print(f"Attention weights shape: {attn_weights.shape}")

    # Apply attention to values
    output = torch.matmul(attn_weights, V)
    print(f"Attention output shape: {output.shape}")

    # Test concatenation (for DRAI integration)
    extra_kv = torch.randn(batch_size, num_heads, 1, head_dim)
    K_extended = torch.cat([K, extra_kv], dim=2)
    V_extended = torch.cat([V, extra_kv], dim=2)
    print(f"Extended K/V shapes (for DRAI): {K_extended.shape}, {V_extended.shape}")

    print("✓ All attention mechanism tests passed")
    return True

def test_multi_head_attention():
    """Test PyTorch's built-in MultiheadAttention."""
    print_header("5. MultiheadAttention Module")

    d_model = 128
    num_heads = 8
    seq_len = 20
    batch_size = 4

    # Create MultiheadAttention module
    mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    # Input: (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # Self-attention
    attn_output, attn_weights = mha(x, x, x, need_weights=True)
    print(f"Input shape: {x.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    print("✓ MultiheadAttention module test passed")
    return True

def test_gradient_computation():
    """Test automatic differentiation and gradients."""
    print_header("6. Gradient Computation")

    # Create a simple computation graph
    x = torch.randn(10, 5, requires_grad=True)
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    # Forward pass
    y = torch.matmul(x, w) + b
    loss = y.sum()

    print(f"Forward pass computed, loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print(f"x.grad shape: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"w.grad shape: {w.grad.shape if w.grad is not None else 'None'}")
    print(f"b.grad shape: {b.grad.shape if b.grad is not None else 'None'}")

    print("✓ Gradient computation test passed")
    return True

def test_transformer_layer():
    """Test a simple transformer layer."""
    print_header("7. Transformer Layer")

    # Transformer encoder layer
    d_model = 128
    nhead = 8
    dim_feedforward = 512

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True
    )

    # Input
    batch_size = 4
    seq_len = 20
    src = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = encoder_layer(src)

    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Transformer layer parameters: {sum(p.numel() for p in encoder_layer.parameters()):,}")

    print("✓ Transformer layer test passed")
    return True

def test_save_load():
    """Test model saving and loading."""
    print_header("8. Model Save/Load")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # Save
    torch.save(model.state_dict(), '/tmp/test_model.pt')
    print("Model saved to /tmp/test_model.pt")

    # Load
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    new_model.load_state_dict(torch.load('/tmp/test_model.pt'))
    print("Model loaded successfully")

    # Verify
    x = torch.randn(3, 10)
    out1 = model(x)
    out2 = new_model(x)

    difference = torch.abs(out1 - out2).max().item()
    print(f"Max difference between outputs: {difference:.10f}")

    if difference < 1e-6:
        print("✓ Model save/load test passed")
        return True
    else:
        print("✗ Model save/load test FAILED")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "█"*60)
    print("  PYTORCH INSTALLATION TEST SUITE FOR PROJECT DUALITY")
    print("█"*60)

    tests = [
        ("Basic Installation", test_basic_installation),
        ("Tensor Operations", test_tensor_operations),
        ("Neural Network Basics", test_neural_network_basics),
        ("Attention Mechanism", test_attention_mechanism),
        ("MultiheadAttention", test_multi_head_attention),
        ("Gradient Computation", test_gradient_computation),
        ("Transformer Layer", test_transformer_layer),
        ("Model Save/Load", test_save_load),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            results.append((name, False))

    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("-"*60)

    if passed == total:
        print("\n🎉 All tests passed! PyTorch is ready for Duality development.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
