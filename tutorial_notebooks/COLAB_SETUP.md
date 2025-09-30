# Running RapidFire AI in Google Colab

This guide explains how to run RapidFire AI in Google Colab with proper port forwarding and tunneling.

## Quick Start

The fastest way to get started is to open our Colab Quickstart notebook:

**[Open in Colab →](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/COLAB_QUICKSTART.ipynb)**

## Manual Setup

If you prefer to set things up manually, follow these steps:

### 1. Enable GPU Runtime

1. Go to **Runtime → Change runtime type**
2. Select **GPU** as the hardware accelerator
3. Click **Save**

### 2. Install RapidFire AI

```python
# Install from PyPI
!pip install rapidfireai

# Initialize (installs dependencies)
!rapidfireai init
```

### 3. Start Services

#### Option A: Native Colab Port Forwarding (Recommended)

This is the easiest and free option. It uses Colab's built-in port forwarding:

```bash
!rapidfireai start --colab
```

This will:
- Start all three services (MLflow, Dispatcher, Frontend)
- Automatically expose them using Colab's native port forwarding
- Open URLs in new tabs for easy access

**Pros:**
- ✅ Free and unlimited
- ✅ No external dependencies
- ✅ No account/token required
- ✅ Secure (authenticated URLs)

**Cons:**
- ⚠️ URLs only accessible to you (not shareable)
- ⚠️ Requires running in Colab

#### Option B: Cloudflare Tunnel (For Sharing)

If you need public URLs to share with team members:

```bash
!rapidfireai start --colab --tunnel cloudflare
```

**Pros:**
- ✅ Completely free
- ✅ No registration required
- ✅ Public URLs (shareable)
- ✅ Good performance

**Cons:**
- ⚠️ Random URLs (can't customize)
- ⚠️ Takes ~5 seconds to set up

#### Option C: ngrok (Advanced)

For custom subdomains and advanced features:

```bash
# Set your ngrok token first
import os
os.environ['RF_NGROK_TOKEN'] = 'your_token_here'

# Start with ngrok
!rapidfireai start --colab --tunnel ngrok
```

**Note**: ngrok free tier only allows 1 tunnel at a time. Paid plans required for 3 simultaneous tunnels ($8/month).

## Python API (Advanced Usage)

For more control, use the Python API directly:

```python
from rapidfireai.utils.colab_helper import (
    is_colab,
    expose_port_native,
    expose_port_iframe,
    setup_cloudflare_tunnel,
    expose_rapidfire_services
)

# Check environment
if is_colab():
    print("✅ Running in Colab")

# Expose a single port
url = expose_port_native(3000, "Frontend Dashboard")

# Or expose all services at once
urls = expose_rapidfire_services(
    method='native',  # or 'cloudflare' or 'ngrok'
    mlflow_port=5002,
    dispatcher_port=8080,
    frontend_port=3000
)

print(urls)
```

## Accessing Services

After starting, you'll see URLs for three services:

1. **Frontend Dashboard** (port 3000)
   - Main UI for viewing experiments
   - Interactive Control Operations
   - Real-time metrics

2. **MLflow UI** (port 5002)
   - Detailed experiment tracking
   - Model artifacts
   - Parameter comparison

3. **Dispatcher API** (port 8080)
   - REST API for programmatic access
   - Used by frontend and backend

## Running Experiments

Once services are running, create experiments as normal:

```python
from rapidfireai import Experiment
from rapidfireai.automl import GridSearch, RFModelConfig

# Create experiment
exp = Experiment("my_colab_experiment")

# Define configuration
config = RFModelConfig(
    trainer_type='SFT',
    training_args={
        'learning_rate': [1e-4, 1e-5],
        'per_device_train_batch_size': [4, 8],
        'num_train_epochs': [3]
    }
)

# Run with Grid Search
grid = GridSearch(configs=[config])
exp.run_fit(
    param_config=grid,
    create_model_fn=your_model_fn,
    train_dataset=train_data,
    eval_dataset=eval_data,
    num_chunks=8,
    seed=42
)

# View results
results = exp.get_results()
print(results)
```

## Environment Variables

Customize behavior with environment variables:

```python
import os

# Change ports
os.environ['RF_MLFLOW_PORT'] = '5003'
os.environ['RF_API_PORT'] = '8081'
os.environ['RF_FRONTEND_PORT'] = '3001'

# Choose tunnel method
os.environ['RF_TUNNEL_METHOD'] = 'cloudflare'  # or 'native' or 'ngrok'

# Set ngrok token (if using ngrok)
os.environ['RF_NGROK_TOKEN'] = 'your_token'

# Set database path
os.environ['RF_DB_PATH'] = '/content/db'
```

## Troubleshooting

### Services Won't Start

Check logs:
```bash
!tail -50 mlflow.log
!tail -50 api.log
!tail -50 frontend.log
```

### Port Conflicts

Change ports:
```python
import os
os.environ['RF_MLFLOW_PORT'] = '5003'
# Then restart services
```

### Native Forwarding Not Working

1. Verify you're in Colab:
   ```python
   from rapidfireai.utils.colab_helper import is_colab
   print(is_colab())  # Should be True
   ```

2. Try alternative tunnel:
   ```bash
   !rapidfireai start --colab --tunnel cloudflare
   ```

### Out of Memory

Colab free tier has limited RAM/VRAM. Try:

1. Use smaller models
2. Reduce batch size
3. Enable gradient checkpointing
4. Use quantization (4-bit, 8-bit)

```python
config = {
    'training_args': {
        'per_device_train_batch_size': 2,  # Reduce
        'gradient_accumulation_steps': 8,  # Increase
        'gradient_checkpointing': True,    # Enable
        'fp16': True                       # Use mixed precision
    }
}
```

### Session Timeout

Colab free tier times out after ~12 hours of inactivity. To prevent:

1. Keep a cell running periodically
2. Use Colab Pro for longer sessions
3. Save checkpoints frequently

## Best Practices

### 1. Save Checkpoints

RapidFire automatically saves checkpoints, but you can also export:

```python
# After training
exp.export_best_model('/content/drive/MyDrive/models/my_model')
```

### 2. Mount Google Drive

Persist data across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

# Use Drive for experiments path
exp = Experiment(
    "my_experiment",
    experiments_path="/content/drive/MyDrive/rapidfire_experiments"
)
```

### 3. Monitor Resource Usage

```python
# Check GPU memory
!nvidia-smi

# Check disk space
!df -h

# Check RAM
!free -h
```

### 4. Use Lite Tutorial Notebooks

For faster iterations in Colab, use the `-lite` versions:
- `rf-tutorial-sft-chatqa-lite.ipynb`
- `rf-tutorial-dpo-alignment-lite.ipynb`
- `rf-tutorial-grpo-mathreasoning-lite.ipynb`

These use smaller models and datasets for quicker experimentation.

## Comparison: Tunneling Methods

| Feature | Native Colab | Cloudflare | ngrok |
|---------|-------------|------------|-------|
| **Cost** | Free | Free | Free (limited) / $8/mo |
| **Setup** | Instant | ~5 seconds | ~3 seconds |
| **Registration** | No | No | Yes (auth token) |
| **Public URLs** | No | Yes | Yes |
| **Simultaneous Tunnels** | Unlimited | 3+ | 1 (free) / 3+ (paid) |
| **Custom Subdomains** | N/A | No (free) | Yes (paid) |
| **Bandwidth Limit** | None | None | Limited (free) |
| **Best For** | Personal use | Team sharing | Custom domains |

## Performance Tips

1. **Use Colab Pro** for:
   - Better GPUs (T4 → A100)
   - Longer session times
   - More RAM/VRAM

2. **Optimize chunk size**:
   ```python
   # More chunks = more concurrent training
   # But more overhead per chunk
   exp.run_fit(..., num_chunks=8)  # Good default
   ```

3. **Enable shared memory**:
   ```python
   import os
   os.environ['USE_SHARED_MEMORY'] = 'true'  # Faster checkpointing
   ```

4. **Use appropriate batch sizes**:
   - T4 (16GB): batch_size=4-8
   - V100 (16GB): batch_size=8-16
   - A100 (40GB): batch_size=16-32

## Security Notes

- Native Colab URLs are authenticated (only you can access)
- Cloudflare/ngrok URLs are public (anyone with URL can access)
- Don't share Colab URLs with sensitive data
- Consider setting up authentication for public tunnels

## Next Steps

After setting up in Colab, explore:

1. **[Full Tutorial Notebooks](https://github.com/RapidFireAI/rapidfireai/tree/main/tutorial_notebooks)**
   - Supervised Fine-Tuning (SFT)
   - Direct Preference Optimization (DPO)
   - Group Relative Policy Optimization (GRPO)

2. **[Interactive Control Operations](https://rapidfire-ai-oss-docs.readthedocs-hosted.com/en/latest/ic_ops.html)**
   - Stop/Resume runs
   - Clone with modifications
   - Warm-start from checkpoints

3. **[AutoML Features](https://rapidfire-ai-oss-docs.readthedocs-hosted.com/en/latest/automl.html)**
   - Grid Search
   - Random Search
   - Custom search algorithms

4. **[API Documentation](https://rapidfire-ai-oss-docs.readthedocs-hosted.com/en/latest/api.html)**
   - Complete API reference
   - Advanced configuration options

## Support

- **Documentation**: https://rapidfire-ai-oss-docs.readthedocs-hosted.com/
- **GitHub Issues**: https://github.com/RapidFireAI/rapidfireai/issues
- **Discord**: [Join our community]
- **Email**: support@rapidfire.ai

---

**Happy experimenting! 🚀**
