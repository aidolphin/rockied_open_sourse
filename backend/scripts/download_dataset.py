#!/usr/bin/env python3
"""
Download the rock classification dataset using multiple methods.
Tries: kagglehub → kaggle CLI → direct download.
"""
import os
import sys
import subprocess
from pathlib import Path
import json

def setup_kaggle_credentials():
    """Setup Kaggle credentials from environment or ~/.kaggle/kaggle.json."""
    if 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
        return True
        
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                creds = json.load(f)
                os.environ['KAGGLE_USERNAME'] = creds['username']
                os.environ['KAGGLE_KEY'] = creds['key']
            return True
        except Exception as e:
            print(f"Error reading Kaggle credentials: {e}")
    return False

def try_kagglehub(dataset_slug):
    """Try downloading using kagglehub."""
    try:
        import kagglehub
        print("📦 Downloading with kagglehub...")
        path = kagglehub.dataset_download(dataset_slug)
        print(f"✅ Downloaded to: {path}")
        return True
    except ImportError:
        print("⚠️ kagglehub not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kagglehub'])
            return try_kagglehub(dataset_slug)  # retry after install
        except Exception as e:
            print(f"❌ Failed to install kagglehub: {e}")
    except Exception as e:
        print(f"❌ kagglehub download failed: {e}")
    return False

def try_kaggle_cli(dataset_slug, download_dir):
    """Try downloading using kaggle CLI."""
    try:
        import kaggle
        print("📦 Downloading with kaggle CLI...")
        subprocess.check_call(['kaggle', 'datasets', 'download', '-d', dataset_slug, 
                             '-p', str(download_dir), '--force'])
        return True
    except ImportError:
        print("⚠️ kaggle CLI not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
            return try_kaggle_cli(dataset_slug, download_dir)  # retry after install
        except Exception as e:
            print(f"❌ Failed to install kaggle CLI: {e}")
    except Exception as e:
        print(f"❌ kaggle CLI download failed: {e}")
    return False

def try_direct_download(dataset_slug, zip_path):
    """Try downloading directly using requests."""
    if not setup_kaggle_credentials():
        print("❌ No Kaggle credentials found")
        return False
        
    try:
        import requests
        print("📦 Attempting direct download...")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
        headers = {
            'Authorization': f"Basic {os.environ['KAGGLE_USERNAME']}:{os.environ['KAGGLE_KEY']}"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded to: {zip_path}")
            return True
        else:
            print(f"❌ Download failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Direct download failed: {e}")
    return False

def main():
    dataset_slug = "salmaneunus/rock-classification"
    if len(sys.argv) > 1:
        dataset_slug = sys.argv[1]
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    zip_path = data_dir / 'rock-classification.zip'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try download methods in sequence
    if not try_kagglehub(dataset_slug):
        if not try_kaggle_cli(dataset_slug, data_dir):
            if not try_direct_download(dataset_slug, zip_path):
                print("❌ All download methods failed")
                print("Please ensure you have Kaggle credentials:")
                print("1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
                print("2. Or place credentials in ~/.kaggle/kaggle.json")
                sys.exit(1)
    
    print("✅ Download completed successfully")

if __name__ == '__main__':
    main()
