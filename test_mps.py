"""
Quick test script to verify MPS (Apple Silicon GPU) is available and working
"""

import torch

print("=" * 80)
print("PyTorch Device Detection Test")
print("=" * 80)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check MPS availability
print("\n1. MPS (Apple Metal) Support:")
if hasattr(torch.backends, 'mps'):
    print(f"   - MPS backend available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"   - MPS built: {torch.backends.mps.is_built()}")
        print("   ✅ MPS is ready to use!")
    else:
        print("   ⚠️  MPS backend not available")
else:
    print("   ⚠️  MPS not supported (PyTorch version too old)")

# Check CUDA availability
print("\n2. CUDA (NVIDIA GPU) Support:")
print(f"   - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - CUDA version: {torch.version.cuda}")
    print(f"   - GPU count: {torch.cuda.device_count()}")

# Determine device (Priority: CUDA > MPS > CPU)
print("\n3. Selected Device (Priority: CUDA > MPS > CPU):")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"   - Device: {device} (NVIDIA CUDA GPU)")
    print(f"   - GPU Name: {torch.cuda.get_device_name(0)}")
    print("   - Status: ✅ GPU acceleration enabled (best for VM/Cloud)")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"   - Device: {device} (Apple Metal)")
    print("   - Status: ✅ GPU acceleration enabled (Apple Silicon)")
else:
    device = torch.device('cpu')
    print(f"   - Device: {device}")
    print("   - Status: ⚠️  CPU only (training will be slower)")

# Test tensor operations on selected device
print("\n4. Testing Tensor Operations:")
try:
    # Create a small tensor
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)

    # Perform operation
    z = torch.matmul(x, y)

    print(f"   - Created tensor on {device}: ✅")
    print(f"   - Matrix multiplication: ✅")
    print(f"   - Result shape: {z.shape}")
    print(f"   - Result device: {z.device}")
    print("\n   🎉 Device is working correctly!")

except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Note: If MPS fails, the code will fall back to CPU")

# Performance estimate
print("\n5. Expected Training Time:")
if device.type == 'mps':
    print("   - With MPS: ~20-30 minutes for full pipeline")
    print("   - Speedup: ~3-5x faster than CPU")
elif device.type == 'cuda':
    print("   - With CUDA: ~15-25 minutes for full pipeline")
    print("   - Speedup: ~4-6x faster than CPU")
else:
    print("   - With CPU: ~1-2 hours for full pipeline")

print("\n" + "=" * 80)
print("Device check complete!")
print("=" * 80)

# Print recommendation
if device.type in ['mps', 'cuda']:
    print("\n✅ You're all set! Your training will use GPU acceleration.")
    print("   Run: python run_complete_pipeline.py")
else:
    print("\n⚠️  GPU not available. Training will use CPU (slower).")
    print("   You can still run the code, it will just take longer.")
    print("   Run: python run_complete_pipeline.py")
