# Use a single GPU to avoid NCCL/DataParallel conflicts when running as script (e.g. CI).
# Worker subprocesses re-import this module before RapidFire sets per-worker GPU, so they
# see all GPUs and Trainer uses DataParallel → NCCL error. Keep this unless RapidFire
# fixes env before any user/torch imports in workers.
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

from datasets import load_dataset

dataset=load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Select a subset of the dataset for demo purposes
train_dataset=dataset["train"].select(range(32))
eval_dataset=dataset["train"].select(range(32,40))
train_dataset=train_dataset.shuffle(seed=42)
eval_dataset=eval_dataset.shuffle(seed=42)

def sample_formatting_function(row):
    """Function to preprocess each example from dataset"""
    # Special tokens for formatting
    SYSTEM_PROMPT = "You are a helpful and friendly customer support assistant. Please answer the user's query to the best of your ability."
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
            
        ],
        "completion": [
            {"role": "assistant", "content": row["response"]}
        ]
    }

def sample_compute_metrics(eval_preds):  
    """Optional function to compute eval metrics based on predictions and labels"""
    predictions, labels = eval_preds

    # Standard text-based eval metrics: Rouge and BLEU
    import evaluate
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    rouge_output = rouge.compute(predictions=predictions, references=labels, use_stemmer=True)
    rouge_l = rouge_output["rougeL"]
    bleu_output = bleu.compute(predictions=predictions, references=labels)
    bleu_score = bleu_output["bleu"]

    return {
        "rougeL": round(rouge_l, 4),
        "bleu": round(bleu_score, 4),
    }

# 2 LoRA PEFT configs lite with different adapter capacities
peft_configs_lite = List([
    RFLoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Standard transformer naming
        bias="none"
    ),
    RFLoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Standard naming
        bias="none"
    )
])

# 2 base models x 2 peft configs = 4 combinations in total
config_set_lite = List([
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B model
        peft_config=peft_configs_lite,
        training_args=RFSFTConfig(
            learning_rate=1e-3,  # Higher LR for very small model
            lr_scheduler_type="linear",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            max_steps=8,
            gradient_accumulation_steps=1,   # No accumulation needed
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=4,
            bf16=True,
        ),
        model_type="causal_lm",
        model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
        formatting_func=sample_formatting_function,
        compute_metrics=sample_compute_metrics,
        generation_config={
            "max_new_tokens": 256,
            "temperature": 0.8,  # Higher temp for tiny model
            "top_p": 0.9,
            "top_k": 30,         # Reduced top_k
            "repetition_penalty": 1.05,
        }
    ),
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B model
        peft_config=peft_configs_lite,
        training_args=RFSFTConfig(
            learning_rate=1e-4,  # Higher LR for very small model
            lr_scheduler_type="linear",
            per_device_train_batch_size=4,  # Even larger batch size
            per_device_eval_batch_size=4,
            max_steps=8,
            gradient_accumulation_steps=1,   # No accumulation needed
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=4,
            bf16=True,
        ),
        model_type="causal_lm",
        model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
        formatting_func=sample_formatting_function,
        compute_metrics=sample_compute_metrics,
        generation_config={
            "max_new_tokens": 256,
            "temperature": 0.8,  # Higher temp for tiny model
            "top_p": 0.9,
            "top_k": 30,         # Reduced top_k
            "repetition_penalty": 1.05,
        }
    )
])


def sample_create_model(model_config): 
     """Function to create model object for any given config; must return tuple of (model, tokenizer)"""
     from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM

     model_name = model_config["model_name"]
     model_type = model_config["model_type"]
     model_kwargs = model_config["model_kwargs"]
 
     if model_type == "causal_lm":
          model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
     elif model_type == "seq2seq_lm":
          model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
     elif model_type == "masked_lm":
          model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
     elif model_type == "custom":
          # Handle custom model loading logic, e.g., loading your own checkpoints
          # model = ... 
          pass
     else:
          # Default to causal LM
          model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
      
     tokenizer = AutoTokenizer.from_pretrained(model_name)
      
     return (model,tokenizer)

# Simple grid search across all sets of config knob values = 4 combinations in total
config_group = RFGridSearch(
    configs=config_set_lite,
    trainer_type="SFT"
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run RapidFire SFT sanity experiment")
    parser.add_argument(
        "--experiment-name",
        default="exp1-chatqa-sanity-101",
        help="Unique experiment name (default: exp1-chatqa-sanity-101)",
    )
    args = parser.parse_args()

    # Every experiment instance must be uniquely named
    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")

    # Launch training of all configs in the config_group with swap granularity of 4 chunks
    experiment.run_fit(config_group, sample_create_model, train_dataset, eval_dataset, num_chunks=4, seed=42)

    # End the experiment
    experiment.end()
