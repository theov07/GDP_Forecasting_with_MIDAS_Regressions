import subprocess
import sys

def install_requirements():
    """Install packages from requirements.txt if not already installed."""
    required_packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'])
        print("✅ Installation complete! Please restart the kernel.")
    else:
        print("✅ All required packages are already installed.")