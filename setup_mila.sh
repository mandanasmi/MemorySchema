#!/bin/bash

# MemorySchema Mila Cluster Setup Script
# This script sets up the MemorySchema repository on the Mila cluster

set -e  # Exit on any error

echo " Setting up MemorySchema on Mila cluster..."

# Check if we're on Mila cluster
if [[ ! "$HOSTNAME" == *"mila"* ]]; then
    echo "⚠️  Warning: This script is designed for the Mila cluster"
    echo "   Current hostname: $HOSTNAME"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set up directory structure
PROJECT_DIR="$HOME/MemorySchema"
REPO_URL="https://github.com/mandanasamiei/MemorySchema.git"  # Update this with your actual repo URL

echo "📁 Setting up project directory: $PROJECT_DIR"

# Clone or update repository
if [ -d "$PROJECT_DIR" ]; then
    echo "📂 Repository already exists, updating..."
    cd "$PROJECT_DIR"
    git pull origin main
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install additional useful packages for Mila cluster
echo " Installing additional packages for Mila cluster..."
pip install jupyter matplotlib seaborn

# Create a simple test script
echo " Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify MemorySchema setup on Mila cluster
"""

import sys
import numpy as np
import gymnasium as gym

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import minigrid
        import pygame
        import numpy as np
        import gymnasium as gym
        print(" All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test that the custom environment can be created"""
    try:
        # Try to import and create environment
        from environment import create_environment
        env = create_environment(size=8)
        obs, info = env.reset()
        print(" Custom environment created and reset successfully")
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    print(" Testing MemorySchema setup on Mila cluster...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {sys.path[0]}")
    
    success = True
    success &= test_imports()
    success &= test_environment()
    
    if success:
        print("\n🎉 Setup test completed successfully!")
        print("You can now run the MemorySchema project on Mila cluster.")
    else:
        print("\n❌ Setup test failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make test script executable
chmod +x test_setup.py

# Run the test
echo " Running setup test..."
python test_setup.py

# Create activation script for easy environment activation
echo "📝 Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for MemorySchema environment on Mila

cd "$HOME/MemorySchema"
source venv/bin/activate
echo "🐍 MemorySchema environment activated!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Installed packages:"
pip list | grep -E "(minigrid|gymnasium|numpy|pygame)"
EOF

chmod +x activate_env.sh

# Create a simple run script
echo "📝 Creating run script..."
cat > run_demo.sh << 'EOF'
#!/bin/bash
# Run MemorySchema demo on Mila cluster

cd "$HOME/MemorySchema"
source venv/bin/activate

echo " Running MemorySchema demo..."
python demo.py
EOF

chmod +x run_demo.sh

echo ""
echo "🎉 MemorySchema setup completed successfully on Mila cluster!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: source ~/MemorySchema/activate_env.sh"
echo "   2. Run the demo: ~/MemorySchema/run_demo.sh"
echo "   3. Or manually: cd ~/MemorySchema && source venv/bin/activate && python demo.py"
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo "🐍 Virtual environment: $PROJECT_DIR/venv"
echo ""
echo " Tips for Mila cluster:"
echo "   - Use 'srun' for interactive sessions: srun --pty bash"
echo "   - Use 'sbatch' for batch jobs"
echo "   - Check available GPUs with: nvidia-smi"
echo "   - Monitor jobs with: squeue -u \$USER"
