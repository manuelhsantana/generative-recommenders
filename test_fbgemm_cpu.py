#!/usr/bin/env python3
"""
Test script to verify fbgemm_gpu CPU operations work correctly.
Run with: conda activate torchRecCPU && python test_fbgemm_cpu.py
"""

import torch
import sys

def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(test_name: str, success: bool, details: str = "") -> None:
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")

def test_torch_basics():
    """Test basic torch functionality on CPU."""
    print_header("Testing PyTorch Basics")
    
    try:
        # Basic tensor ops
        x = torch.randn(10, 5, device="cpu")
        y = torch.matmul(x, x.T)
        print_result("Basic tensor operations", True, f"Shape: {y.shape}")
        
        # Check CUDA availability (should be False for CPU-only)
        cuda_available = torch.cuda.is_available()
        print_result("CUDA availability check", True, f"CUDA available: {cuda_available}")
        
        return True
    except Exception as e:
        print_result("Basic tensor operations", False, str(e))
        return False

def test_fbgemm_import():
    """Test fbgemm_gpu import."""
    print_header("Testing FBGEMM Import")
    
    try:
        import fbgemm_gpu
        print_result("Import fbgemm_gpu", True)
        return True
    except ImportError as e:
        print_result("Import fbgemm_gpu", False, str(e))
        return False

def test_jagged_to_padded_dense():
    """Test torch.ops.fbgemm.jagged_to_padded_dense on CPU."""
    print_header("Testing jagged_to_padded_dense")
    
    try:
        # Create jagged tensor data
        # Sequence lengths: [3, 2, 4] -> total 9 elements
        values = torch.randn(9, 8, device="cpu")  # 9 elements, 8 features each
        offsets = torch.tensor([0, 3, 5, 9], dtype=torch.int64, device="cpu")
        max_length = 5
        padding_value = 0.0
        
        # Call the fbgemm operation
        result = torch.ops.fbgemm.jagged_to_padded_dense(
            values=values,
            offsets=[offsets],
            max_lengths=[max_length],
            padding_value=padding_value,
        )
        
        expected_shape = (3, max_length, 8)  # (batch, max_len, features)
        success = result.shape == expected_shape
        print_result(
            "jagged_to_padded_dense", 
            success, 
            f"Output shape: {result.shape}, Expected: {expected_shape}"
        )
        return success
    except Exception as e:
        print_result("jagged_to_padded_dense", False, str(e))
        return False

def test_dense_to_jagged():
    """Test torch.ops.fbgemm.dense_to_jagged on CPU."""
    print_header("Testing dense_to_jagged")
    
    try:
        # Create dense tensor
        batch_size = 3
        max_length = 5
        features = 8
        dense = torch.randn(batch_size, max_length, features, device="cpu")
        
        # Offsets define actual lengths: [3, 2, 4]
        offsets = torch.tensor([0, 3, 5, 9], dtype=torch.int64, device="cpu")
        
        # Call the fbgemm operation
        result, _ = torch.ops.fbgemm.dense_to_jagged(
            dense=dense,
            offsets=[offsets],
            total_L=None,
        )
        
        expected_total = 9  # 3 + 2 + 4
        success = result.shape[0] == expected_total and result.shape[1] == features
        print_result(
            "dense_to_jagged",
            success,
            f"Output shape: {result.shape}, Expected total elements: {expected_total}"
        )
        return success
    except Exception as e:
        print_result("dense_to_jagged", False, str(e))
        return False

def test_jagged_dense_elementwise_add():
    """Test torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output on CPU."""
    print_header("Testing jagged_dense_elementwise_add_jagged_output")
    
    try:
        # Jagged tensor: 9 total elements across 3 sequences
        jagged_values = torch.randn(9, 8, device="cpu")
        offsets = torch.tensor([0, 3, 5, 9], dtype=torch.int64, device="cpu")
        
        # Dense tensor: (batch, max_len, features)
        max_length = 5
        dense = torch.randn(3, max_length, 8, device="cpu")
        
        # Call the fbgemm operation
        result = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values=jagged_values,
            x_offsets=[offsets],
            y=dense,
        )
        
        success = result.shape == jagged_values.shape
        print_result(
            "jagged_dense_elementwise_add_jagged_output",
            success,
            f"Output shape: {result.shape}, Expected: {jagged_values.shape}"
        )
        return success
    except Exception as e:
        print_result("jagged_dense_elementwise_add_jagged_output", False, str(e))
        return False

def test_torchrec_import():
    """Test torchrec import and basic functionality."""
    print_header("Testing TorchRec Import")
    
    try:
        import torchrec
        print_result("Import torchrec", True)
        
        # Test KeyedJaggedTensor creation
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        
        kjt = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64),
            lengths=torch.tensor([2, 1, 1, 2], dtype=torch.int64),
        )
        print_result("KeyedJaggedTensor creation", True, f"Keys: {kjt.keys()}")
        
        return True
    except Exception as e:
        print_result("TorchRec operations", False, str(e))
        return False

def test_distributed_gloo():
    """Test torch.distributed with gloo backend."""
    print_header("Testing Distributed (Gloo Backend)")
    
    try:
        import torch.distributed as dist
        
        # Check if gloo is available
        gloo_available = dist.is_gloo_available()
        print_result("Gloo backend available", gloo_available)
        
        if not gloo_available:
            return False
            
        return True
    except Exception as e:
        print_result("Distributed gloo check", False, str(e))
        return False

def test_embedding_bag_collection():
    """Test TorchRec EmbeddingBagCollection on CPU."""
    print_header("Testing EmbeddingBagCollection (CPU)")
    
    try:
        from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        
        # Create embedding config
        ebc_config = [
            EmbeddingBagConfig(
                name="table_0",
                embedding_dim=16,
                num_embeddings=100,
                feature_names=["feature_0"],
            ),
            EmbeddingBagConfig(
                name="table_1", 
                embedding_dim=16,
                num_embeddings=100,
                feature_names=["feature_1"],
            ),
        ]
        
        # Create EmbeddingBagCollection
        ebc = EmbeddingBagCollection(
            tables=ebc_config,
            device=torch.device("cpu"),
        )
        
        # Create input KeyedJaggedTensor
        kjt = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64),
            lengths=torch.tensor([2, 1, 1, 2], dtype=torch.int64),
        )
        
        # Forward pass
        output = ebc(kjt)
        
        print_result(
            "EmbeddingBagCollection forward",
            True,
            f"Output keys: {list(output.keys())}"
        )
        return True
    except Exception as e:
        print_result("EmbeddingBagCollection", False, str(e))
        return False

def main():
    print("\n" + "="*60)
    print(" FBGEMM CPU Compatibility Test Suite")
    print(" For DLRM v3 CPU Training with Gloo Backend")
    print("="*60)
    
    results = {}
    
    # Run all tests
    results["torch_basics"] = test_torch_basics()
    results["fbgemm_import"] = test_fbgemm_import()
    
    if results["fbgemm_import"]:
        results["jagged_to_padded_dense"] = test_jagged_to_padded_dense()
        results["dense_to_jagged"] = test_dense_to_jagged()
        results["jagged_dense_elementwise_add"] = test_jagged_dense_elementwise_add()
    else:
        results["jagged_to_padded_dense"] = False
        results["dense_to_jagged"] = False
        results["jagged_dense_elementwise_add"] = False
    
    results["torchrec_import"] = test_torchrec_import()
    results["distributed_gloo"] = test_distributed_gloo()
    results["embedding_bag_collection"] = test_embedding_bag_collection()
    
    # Summary
    print_header("Test Summary")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Ready for CPU training.")
        return 0
    else:
        print("\n  ⚠️  Some tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
