import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Training helpers
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model

from utils import calculate_mape, format_time_difference
from finance.meta_prompt import finance_mse_metaprompt_generation

def load_timeseries_dataset(df):
    df["text"] = df["text"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    for col in ["input_window", "output_window", "input_timestamps"]:
        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "filename": row.get("filename", ""),
            "text": row["text"]["content"],
            "input_window": row["input_window"],
            "output_window": row["output_window"],
            "input_timestamps": row["input_timestamps"],
        })
    return samples

def build_hf_dataset(raw_data, tokenizer, mode="combined"):
    def tokenize(sample):
        input_ts = sample["input_window"]
        output_ts = sample["output_window"]
        timestamps = sample["input_timestamps"]
        datetime_list = [
            datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
            for s in timestamps
        ]
        granularity = format_time_difference(timestamps[1] - timestamps[0])

        prompt = finance_mse_metaprompt_generation(
            text=sample["text"],
            prices=input_ts,
            start_datetime=datetime_list[0],
            end_datetime=datetime_list[-1],
            pred_end_datetime=datetime_list[-1],
            granularity=granularity,
            prediction_length=len(output_ts),
            mode=mode
        )

        target = ", ".join([f"{x:.4f}" for x in output_ts]) # Keep the number of decimals to 4 (uniform) for generation stability
        full_text = prompt + target + tokenizer.eos_token
        inputs = tokenizer(full_text, truncation=True, max_length=tokenizer.model_max_length)
        labels = inputs.input_ids.copy()

        # Mask the prompt portion of the input so loss is only on the numeric sequence
        prompt_ids = tokenizer(prompt, truncation=True, max_length=tokenizer.model_max_length).input_ids
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        inputs["labels"] = labels
        return inputs

    dataset = Dataset.from_list(raw_data)
    return dataset.map(tokenize, remove_columns=dataset.column_names)

def parse_series(s):
    try:
        return list(map(float, s.strip().split(",")))
    except:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Save path for fine-tuned model")
    parser.add_argument("--mode", type=str, default="combined", help="Prompt mode: combined, text_only, or timeseries_only")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model")
    args = parser.parse_args()

    print("Loading dataset")
    raw_data = load_timeseries_dataset(args.dataset_path)

    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Applying LoRA")  
    lora_config = LoraConfig(
        lora_alpha=32, # moderate scaling factor of 32 / 8 = 4
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("Preprocessing dataset")
    total_len = len(raw_data)
    train_end = int(0.8 * total_len)
    val_end = int(0.9 * total_len)
    train_data = raw_data[:train_end]
    val_data = raw_data[train_end:val_end]
    test_data = raw_data[val_end:]

    train_dataset = build_hf_dataset(train_data, tokenizer, args.mode)
    val_dataset = build_hf_dataset(val_data, tokenizer, args.mode)
    test_dataset = build_hf_dataset(test_data, tokenizer, args.mode)

    print("Starting training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1, # as we are using long prompts
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, # effective batch size: 1 * 4 = 4
        num_train_epochs=3,
        logging_steps=20,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        bf16=True,
        logging_dir=Path(args.output_dir) / "logs",
        load_best_model_at_end=True,
    ) 

    # Note: In this iteration, we treat numeric forecasting as a language modeling problem.
    # The model is trained to generate comma-separated numerical sequences as text using standard cross-entropy loss.
    # We intentionally do not use regression losses (e.g. MSE, MAE, MAPE) during training,since the model does not predict continuous values.
    # Regression metrics are instead computed at evaluation time, after parsing the text output into floats.
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained(Path(args.output_dir) / "final")

    print("Evaluating on MSE, MAE, RMSE, MAPE")
    predictions = trainer.predict(val_dataset)
    preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

    y_pred = [parse_series(p) for p in preds]
    y_true = [parse_series(t) for t in labels]

    metrics = {"mse": [], "mae": [], "rmse": [], "mape": []}
    for gt, pr in zip(y_true, y_pred):
        if len(gt) != len(pr):
            continue
        gt_arr = np.array(gt)
        pr_arr = np.array(pr)
        metrics["mse"].append(np.mean((gt_arr - pr_arr) ** 2))
        metrics["mae"].append(np.mean(np.abs(gt_arr - pr_arr)))
        metrics["rmse"].append(np.sqrt(np.mean((gt_arr - pr_arr) ** 2)))
        metrics["mape"].append(calculate_mape(gt, pr))

    print("Evaluation Summary:")
    for key, vals in metrics.items():
        print(f"{key.upper()}: {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()