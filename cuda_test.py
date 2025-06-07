"""CUDA Diagnostic Script for TranscribeAI
Run this to check if your PyTorch CUDA setup is working properly.
"""

import torch
import sys

def test_cuda_setup():
    print("=== CUDA Diagnostic Report ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n1. CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
        
        # Device information
        device_count = torch.cuda.device_count()
        print(f"\n2. CUDA Devices: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"   Device {i}: {props.name}")
            print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
        
        # Memory test
        print(f"\n3. Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name()}")
        
        # Test tensor operations
        print(f"\n4. Testing CUDA Operations:")
        try:
            # Create a tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"   ✓ Basic GPU tensor operations work")
            print(f"   ✓ Result tensor device: {z.device}")
            
            # Test memory allocation
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   ✓ Memory allocated: {memory_allocated:.1f} MB")
            print(f"   ✓ Memory reserved: {memory_reserved:.1f} MB")
            
        except Exception as e:
            print(f"   ✗ GPU operations failed: {e}")
        
        # Test model loading simulation
        print(f"\n5. Testing Model Loading:")
        try:
            # Simulate model parameters
            model_size_mb = 244  # Approximate size of whisper-small in MB
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   Model size (whisper-small): ~{model_size_mb} MB")
            print(f"   Available GPU memory: {available_memory_gb:.1f} GB")
            
            if available_memory_gb * 1024 > model_size_mb * 2:  # 2x safety margin
                print(f"   ✓ Sufficient memory for whisper-small model")
            else:
                print(f"   ⚠ May have insufficient memory for larger models")
                
        except Exception as e:
            print(f"   ✗ Memory check failed: {e}")
            
    else:
        print("\n   CUDA not available. Possible reasons:")
        print("   - PyTorch was installed without CUDA support")
        print("   - NVIDIA drivers are not installed or outdated")
        print("   - CUDA toolkit version mismatch")
        print("   - No compatible NVIDIA GPU found")
        
        # Check if PyTorch was built with CUDA
        print(f"\n   PyTorch built with CUDA: {torch.version.cuda is not None}")
        if torch.version.cuda:
            print(f"   PyTorch CUDA version: {torch.version.cuda}")
        else:
            print("   → Install PyTorch with CUDA support:")
            print("     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    print(f"\n=== End Diagnostic Report ===")

if __name__ == "__main__":
    test_cuda_setup() 