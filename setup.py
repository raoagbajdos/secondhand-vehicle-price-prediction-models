#!/usr/bin/env python3
"""
Setup script for the car price prediction project.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command: str, description: str):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Setup the project environment."""
    print("🚀 Setting up Second-Hand Car Price Prediction Project")
    
    # Check if uv is installed
    print("\n📋 Checking dependencies...")
    
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV package manager found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV package manager not found")
        print("Please install UV first: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    # Create virtual environment
    if not run_command("uv venv", "Creating virtual environment"):
        return False
    
    # Install dependencies
    if not run_command("uv pip install -e \".[dev]\"", "Installing dependencies"):
        print("⚠️  Some dependencies might not be available. The project structure is still created.")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from template")
        else:
            print("⚠️  .env.example not found, please create .env manually")
    
    # Setup pre-commit hooks
    try:
        run_command("uv pip install pre-commit", "Installing pre-commit")
        run_command("pre-commit install", "Setting up pre-commit hooks")
    except:
        print("⚠️  Pre-commit setup failed, continuing without it")
    
    print("\n🎉 Setup completed!")
    print("\n📝 Next steps:")
    print("1. Activate the virtual environment:")
    print("   source .venv/bin/activate  # On macOS/Linux")
    print("   .venv\\Scripts\\activate     # On Windows")
    print("\n2. Edit .env with your configuration")
    print("\n3. Start scraping data:")
    print("   scrape-data --brands mercedes,ford,toyota")
    print("\n4. Train models:")
    print("   train-regression --brands mercedes")
    print("\n5. Run the example pipeline:")
    print("   python scripts/run_pipeline.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
