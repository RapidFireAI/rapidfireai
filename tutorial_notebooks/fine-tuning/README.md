# Fine-Tuning Tutorial Notebooks

This directory contains tutorial notebooks demonstrating how to fine-tune language models using RapidFire AI.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `rf-tutorial-sft-trackio.ipynb` | SFT with Trackio experiment tracking (recommended for local development) |
| `rf-tutorial-sft-chatqa.ipynb` | SFT on ChatQA dataset with MLflow tracking |
| `rf-tutorial-sft-chatqa-lite.ipynb` | Lightweight version of ChatQA tutorial |
| `rf-colab-tensorboard-tutorial.ipynb` | SFT with TensorBoard tracking (Google Colab compatible) |

---

## Running rf-tutorial-sft-trackio.ipynb Locally

This guide walks you through running the SFT Trackio tutorial notebook on a local Ubuntu machine with GPU.

### Prerequisites

#### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 7.x or 8.x (e.g., RTX 30xx, RTX 40xx, A100, L4)
- **GPU Memory**: At least 8GB VRAM (16GB+ recommended)
- **RAM**: At least 16GB system memory

#### Software Requirements
- Ubuntu 20.04+ (or compatible Linux distribution)
- Python 3.12.x
- NVIDIA drivers with CUDA support

### Step 1: Clone the Repository

```bash
git clone https://github.com/RapidFireAI/rapidfireai.git
cd rapidfireai
```

### Step 2: Create Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install rapidfireai in development mode
```

### Step 4: Configure HuggingFace Authentication

The notebook downloads models from HuggingFace Hub. You need to authenticate:

```bash
# Install HuggingFace CLI
pip install "huggingface-hub[cli]"

# Login with your token (get one from https://huggingface.co/settings/tokens)
huggingface-cli login

# IMPORTANT: Uninstall hf-xet to avoid known issues
pip uninstall -y hf-xet
```

**Storing your token securely (optional):**
```bash
# Create a .env file (already in .gitignore)
echo 'HF_TOKEN=your_token_here' > .env

# Then authenticate using:
source .env && huggingface-cli login --token $HF_TOKEN
```

### Step 5: Install Node.js (for frontend, optional)

If you want to run the full RapidFire services including the web frontend:

```bash
# Using nvm (recommended for non-root users)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install 22
```

### Step 6: Start RapidFire Services

The notebook requires the RapidFire dispatcher service:

```bash
# Option A: Start just the dispatcher (minimal setup for this notebook)
cd rapidfireai/fit/dispatcher
mkdir -p ~/db
PYTHONPATH="/path/to/rapidfireai:$PYTHONPATH" gunicorn -c gunicorn.conf.py &

# Option B: Start all services (dispatcher, MLflow, frontend)
./setup/fit/start_dev.sh start
```

### Step 7: Launch Jupyter Notebook

```bash
cd /path/to/rapidfireai
source .venv/bin/activate
jupyter notebook tutorial_notebooks/fine-tuning/rf-tutorial-sft-trackio.ipynb
```

Or start Jupyter server for remote access:
```bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

### Step 8: Run the Notebook

1. Open the notebook in your browser
2. Click **"Restart the kernel and run all cells"** (⏩ button)
3. Wait for training to complete (approximately 15-30 minutes for 4 configurations)

### Step 9: View Training Metrics with Trackio

While training is running (or after completion), view the metrics dashboard:

```bash
cd /path/to/rapidfireai
source .venv/bin/activate
trackio show --project "exp1-sft-trackio-demo"
```

This opens a web dashboard (usually at http://localhost:7860) showing:
- Training loss curves for each run
- Evaluation metrics (ROUGE-L, BLEU) over time
- Hyperparameter comparisons across runs

---

## Troubleshooting

### "File Save Error: Failed to fetch"
This error occurs when the Jupyter server crashes or loses connection. Restart the Jupyter server:
```bash
# Kill any existing Jupyter processes
pkill -f jupyter

# Restart
jupyter notebook tutorial_notebooks/fine-tuning/rf-tutorial-sft-trackio.ipynb
```

### "trackio: command not found"
Make sure you've activated the virtual environment:
```bash
source .venv/bin/activate
trackio show --project "exp1-sft-trackio-demo"
```

### GPU Memory Issues
If you encounter CUDA out of memory errors:
1. Reduce `per_device_train_batch_size` in the notebook (e.g., from 4 to 2)
2. Close other GPU-consuming applications
3. Try using a smaller model

### HuggingFace Download Issues
If model downloads fail:
```bash
# Ensure you're logged in
huggingface-cli whoami

# Re-authenticate if needed
huggingface-cli login

# Remove hf-xet if present (known to cause issues)
pip uninstall -y hf-xet
```

---

## What the Notebook Does

This tutorial demonstrates:

1. **Configuring Trackio** as the standalone tracking backend for RapidFire AI
2. **Loading a dataset** (Bitext customer support chatbot training data)
3. **Defining multiple training configurations** using RapidFire's grid search:
   - 2 LoRA configurations (r=8 vs r=32)
   - 2 learning rates (1e-3 vs 1e-4)
   - Total: 4 training runs
4. **Running multi-config training** with automatic metric logging
5. **Viewing experiment metrics** using the Trackio dashboard

The notebook uses TinyLlama-1.1B as the base model for fast iteration during development.

---

## Additional Resources

- [RapidFire AI Documentation](https://oss-docs.rapidfire.ai/)
- [Trackio GitHub Repository](https://github.com/gradio-app/trackio)
- [RapidFire AI Discord](https://discord.gg/6vSTtncKNN)

