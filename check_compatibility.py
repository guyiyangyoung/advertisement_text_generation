#!/usr/bin/env python3
"""
Compatibility checker for Qwen fine-tuning environment
"""

import sys
import pkg_resources
import importlib
from packaging import version

def check_package_version(package_name, min_version=None, max_version=None):
    """Check if a package is installed and meets version requirements"""
    try:
        pkg = pkg_resources.get_distribution(package_name)
        installed_version = pkg.version
        
        print(f"✅ {package_name}: {installed_version}", end="")
        
        if min_version and version.parse(installed_version) < version.parse(min_version):
            print(f" ❌ (requires >= {min_version})")
            return False
        elif max_version and version.parse(installed_version) > version.parse(max_version):
            print(f" ⚠️  (may have compatibility issues with > {max_version})")
            return True
        else:
            if min_version:
                print(f" ✅ (>= {min_version})")
            else:
                print()
            return True
            
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package_name}: Not installed")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: Error checking version - {e}")
        return False

def check_transformers_compatibility():
    """Check transformers version and TrainingArguments compatibility"""
    try:
        from transformers import TrainingArguments
        import inspect
        
        # Check if eval_strategy parameter exists
        sig = inspect.signature(TrainingArguments.__init__)
        params = list(sig.parameters.keys())
        
        if 'eval_strategy' in params:
            print("✅ TrainingArguments: Uses 'eval_strategy' (modern)")
            return 'eval_strategy'
        elif 'evaluation_strategy' in params:
            print("⚠️  TrainingArguments: Uses 'evaluation_strategy' (older)")
            return 'evaluation_strategy'
        else:
            print("❌ TrainingArguments: Cannot determine evaluation parameter")
            return None
            
    except Exception as e:
        print(f"❌ TrainingArguments compatibility check failed: {e}")
        return None

def check_cuda_availability():
    """Check CUDA availability and GPU information"""
    try:
        import torch
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"✅ CUDA: Available with {num_gpus} GPU(s)")
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
        else:
            print("⚠️  CUDA: Not available (will use CPU)")
            return False
            
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

def main():
    """Main compatibility check"""
    print("🔍 Qwen Fine-tuning Environment Compatibility Check")
    print("=" * 55)
    
    # Python version check
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    print("\n📦 Package Versions:")
    print("-" * 25)
    
    # Check critical packages
    packages_ok = True
    
    # Core packages
    packages_ok &= check_package_version("torch", "2.0.0")
    packages_ok &= check_package_version("transformers", "4.30.0")
    packages_ok &= check_package_version("datasets", "2.0.0")
    packages_ok &= check_package_version("peft", "0.4.0")
    packages_ok &= check_package_version("accelerate", "0.20.0")
    
    # Optional but recommended
    check_package_version("bitsandbytes", "0.39.0")
    check_package_version("deepspeed", "0.9.0")
    check_package_version("wandb")
    
    # Data processing
    packages_ok &= check_package_version("pandas", "1.3.0")
    packages_ok &= check_package_version("numpy", "1.20.0")
    
    print("\n🔧 Framework Compatibility:")
    print("-" * 28)
    
    # Check transformers compatibility
    eval_param = check_transformers_compatibility()
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    print("\n📋 Summary:")
    print("-" * 11)
    
    if packages_ok:
        print("✅ All required packages are installed with compatible versions")
    else:
        print("❌ Some packages need to be updated or installed")
    
    if eval_param == 'eval_strategy':
        print("✅ Training scripts are compatible with your transformers version")
    elif eval_param == 'evaluation_strategy':
        print("⚠️  Need to update training scripts for your transformers version")
        print("   Run: sed -i 's/eval_strategy/evaluation_strategy/g' *.py")
    else:
        print("❌ TrainingArguments compatibility issue detected")
    
    if cuda_available:
        print("✅ GPU training is available")
    else:
        print("⚠️  Only CPU training available (will be very slow)")
    
    print("\n🚀 Recommended Actions:")
    print("-" * 22)
    
    if not packages_ok:
        print("1. Update packages: pip install -r requirements.txt --upgrade")
    
    if eval_param == 'evaluation_strategy':
        print("2. Training scripts need parameter name update")
        print("   The scripts have been updated to use 'eval_strategy'")
    
    if not cuda_available:
        print("3. Consider using a GPU for faster training")
    
    print("\n✅ Environment check complete!")
    
    return packages_ok and eval_param is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)