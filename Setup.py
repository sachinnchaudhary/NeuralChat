"""This is file where we will import all requirement libraries so we can share across all files with ease."""

#cell 1.................................................................................
%%writefile "/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup/requirements.txt"
torch==2.3.0
transformers
datasets
accelerate
wandb
sentencepiece
bitsandbytes

#cell 2.................................................................................
%%writefile "/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup/install.sh"  

#cell 3.................................................................................
!bash "/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup/install.sh"  

#cell 4.................................................................................
%%writefile "/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup/init.py"
import torch, os, subprocess
from google.colab import drive

def setup_env():
    # Mount Google Drive (persistent storage)
    drive.mount('/content/drive', force_remount=True)

    # Define project root
    ROOT = "/content/drive/MyDrive/Colab Notebooks/NeuralChat"

    # Run your cached install script from Drive
    subprocess.run(["bash", f"{ROOT}/setup/install.sh"], check=True)

    # Check CUDA and device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Environment ready on", DEVICE, torch.cuda.get_device_name(0) if DEVICE=="cuda" else "CPU")

    return ROOT, DEVICE
