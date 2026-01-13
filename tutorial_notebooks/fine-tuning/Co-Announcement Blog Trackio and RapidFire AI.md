# **Announcing RapidFire AI's Integration with Trackio: Free, Open-Source Experiment Tracking**

## **TL;DR**

We at RapidFire AI are excited to announce our integration with [Trackio](https://github.com/gradio-app/trackio), the free, open-source experiment tracking library from Hugging Face. With this integration, ML engineers can now track, compare, and debug RapidFire experiments using Trackio's local-first dashboard—no server setup required.

This post, co-authored with the Hugging Face team, explains how the integration works and why we chose Trackio for experiment observability.

## **The Problem with Today's LLM Experimentation Workflows**

Most LLM experimentation today is still fragmented and sequential. Engineers typically modify one parameter at a time, rerun pipelines, and then inspect results after the fact. Evaluation tools surface metrics, but they sit downstream of the actual experimentation logic.

As a result:

* Experimentation cycles are slow and manual
* Comparing configurations consistently is difficult
* Experimentation logic ends up fragmented across scripts and notebooks
* Optimization efforts focus on one metric at a time

As LLM systems grow more complex—especially with RAG and post-training workflows—this approach becomes increasingly limiting.

## **Why We Chose Trackio**

When building RapidFire AI's experiment tracking capabilities, we evaluated several options. We chose [Trackio](https://github.com/gradio-app/trackio) for several reasons:

**Free and Open Source**: Trackio is completely free to use and MIT-licensed. There are no usage limits, no premium tiers, and no vendor lock-in. The entire codebase is open for inspection and contribution.

**Local-First Design**: Trackio runs entirely on your machine by default. There's no server to set up, no accounts to create, and no data leaving your environment. Metrics are stored in a local SQLite database that persists between sessions.

**Simple, Familiar API**: Trackio's API consists of just three functions—`init`, `log`, and `finish`. If you've used wandb before, you'll feel right at home:

```python
import trackio

trackio.init(project="my-experiment", config={"lr": 0.001})
trackio.log({"loss": 0.5, "accuracy": 0.92})
trackio.finish()
```

**wandb Compatibility**: Trackio is designed as a drop-in replacement for wandb. You can often switch by simply changing your import:

```python
import trackio as wandb  # Drop-in replacement
```

**Beautiful Gradio Dashboard**: View your experiments with a clean, interactive dashboard by running `trackio show` in your terminal. Compare runs side-by-side, visualize loss curves, and analyze hyperparameter impacts.

## **How RapidFire AI Integrates with Trackio**

RapidFire AI is an experiment execution framework for LLM fine-tuning and post-training that enables hyper-parallel training across multiple configurations. When you run experiments with RapidFire AI, you often have many runs executing simultaneously—each with different hyperparameters, LoRA configurations, or dataset variants.

This is where Trackio becomes invaluable. RapidFire AI has built native Trackio support into its metric logging system:

* **Automatic Metric Capture**: Training metrics (loss, learning rate, gradient norms) are automatically logged to Trackio during training
* **Evaluation Metrics**: Custom evaluation metrics (ROUGE, BLEU, accuracy) are captured at each evaluation step
* **Hyperparameter Tracking**: Each run's configuration is logged, making it easy to understand what parameters produced which results
* **Real-Time Dashboard**: View all your parallel runs in Trackio's dashboard as they train

The integration means you can focus on defining experiments while both tools handle their respective roles: RapidFire AI manages execution, and Trackio provides observability.

## **Practical Workflows**

### **Fine-Tuning Experiments**

When fine-tuning LLMs, you often want to compare multiple configurations:

* Different LoRA ranks and adapter settings
* Various learning rates and schedulers
* Multiple base models or dataset variants

RapidFire AI executes these configurations in parallel, and Trackio captures every metric. In the dashboard, you can immediately see which configurations are converging faster, which have lower loss, and which produce better evaluation scores.

### **RAG Configuration Comparison**

For RAG systems, the integration helps you systematically compare:

* Chunk sizes and overlap strategies
* Embedding model choices
* Retrieval and reranking approaches

Each configuration's retrieval metrics, latency, and accuracy are logged to Trackio, making tradeoffs visible at a glance.

### **Using Trackio Insights with RapidFire's IC Ops**

RapidFire AI includes Interactive Control Operations (IC Ops) that let you control experiments in real-time. Combined with Trackio's observability:

1. **Monitor in Trackio**: Watch your parallel runs in the Trackio dashboard
2. **Identify underperformers**: Spot runs with poor loss curves or diverging metrics
3. **Take action with IC Ops**: Stop underperforming runs to free up GPU resources, or clone promising configurations with modified parameters

This workflow turns experiment tracking from passive observation into active experiment management.

## **What Gets Tracked**

When you run RapidFire AI experiments with Trackio enabled, the following metrics are automatically captured:

* **Training metrics**: loss, learning_rate, grad_norm, epoch, step
* **Evaluation metrics**: eval_loss, plus any custom metrics you define (e.g., rougeL, bleu)
* **Configuration**: All hyperparameters for each run
* **Run metadata**: Timestamps, run names, experiment grouping

In the Trackio dashboard, you can:

* Compare loss curves across all runs
* Filter and group runs by hyperparameters
* Zoom into specific training phases
* Export data for further analysis

## **Get Started**

Ready to try the integration? Here's how to get started:

**Install both packages:**

```bash
pip install rapidfireai trackio
```

**Try the tutorial notebook:**

We've created a hands-on tutorial that walks through a complete fine-tuning experiment with Trackio tracking. The notebook demonstrates configuring Trackio, running parallel experiments, and viewing results in the dashboard.

👉 [RapidFire AI + Trackio Tutorial Notebook](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-trackio.ipynb)

**Learn more:**

* [Trackio GitHub Repository](https://github.com/gradio-app/trackio) - Full documentation and examples
* [Trackio Documentation](https://huggingface.co/docs/trackio/index) - API reference and guides
* [RapidFire AI Documentation](https://oss-docs.rapidfire.ai/) - Getting started with RapidFire AI
* [RapidFire AI GitHub](https://github.com/RapidFireAI/rapidfireai) - Source code and more tutorials

## **Conclusion**

By integrating Trackio into RapidFire AI, we've combined hyper-parallel experiment execution with free, open-source experiment tracking. ML engineers can now run many configurations simultaneously while maintaining full visibility into every run's progress.

We believe experiment tracking should be accessible to everyone—not locked behind pricing tiers or requiring complex infrastructure. Trackio embodies this philosophy, and we're excited to bring it to the RapidFire AI community.

We invite you to try the integration, share feedback, and help shape the future of LLM experimentation tooling.

---

*This post was co-authored by the RapidFire AI team and Hugging Face contributors.*
