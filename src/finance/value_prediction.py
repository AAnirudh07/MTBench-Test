import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
# sys.path.append("../..")
from tqdm import tqdm
import pandas as pd

from models.model_factory import ModelFactory
from meta_prompt import (
    finance_bb_metaprompt_generation,
    finance_macd_metaprompt_generation,
    finance_mse_metaprompt_generation,
    parse_val_prediction_response,
)
from utils import (
    save_to_json,
    calculate_mape,
    format_time_difference,
    plot_series,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="path to the dataset")
parser.add_argument("--save_path", type=str, help="path to save the results")
parser.add_argument("--indicator", default="macd", type=str, help="macd, bb, or time")
parser.add_argument("--model_type",  type=str, help="deepseek or llama")
parser.add_argument("--model",  type=str, help="model name")
parser.add_argument(
    "--mode",
    type=str,
    default="combined",
    help="choose from timeseries_only, text_only, combined",
)
args = parser.parse_args()

save_path = Path(args.save_path)
details_path = save_path / "output_details"
visualizations_path = save_path / "visualizations"
details_path.mkdir(parents=True, exist_ok=True)
visualizations_path.mkdir(parents=True, exist_ok=True)

data_list = []
df = pd.read_parquet(args.dataset_path)
filename = Path(args.dataset_path).name

df["text"] = df["text"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
df["technical"] = df["technical"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

for col in ["input_window", "output_window", "input_timestamps"]:
    df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

for _, row in df.iterrows():
    text = row["text"]
    technical = row["technical"]

    extracted_data = {
        "filename": filename,
        "input_window": row["input_window"],
        "output_window": row["output_window"],
        "text": text["content"],
        "input_timestamps": row["input_timestamps"],
        "in_macd": technical.get("in_macd"),
        "out_macd": technical.get("out_macd"),
        "in_upper_bb": technical.get("in_upper_bb"),
        "out_upper_bb": technical.get("out_upper_bb"),
    }

    data_list.append(extracted_data)

model = ModelFactory.get_model(model_type=args.model_type, model_name=args.model)

result_list = []
tot_samples = len(data_list)
print(f"Evaluating {tot_samples} samples...")

epoch_results = []
cumulative_mse, cumulative_mae, cumulative_rmse, cumulative_mape = [], [], [], []
for idx, sample in tqdm(enumerate(data_list), total=tot_samples):
    try:
        datetime_list = [
            datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")
            for s in sample["input_timestamps"]
        ]
        text = sample["text"]
        input_ts = sample["input_window"]
        granularity_string = format_time_difference(
            sample["input_timestamps"][1] - sample["input_timestamps"][0]
        )

        if args.indicator == "macd":
            output_ts = sample["out_macd"]
            designed_prompt = finance_macd_metaprompt_generation(
                text=text,
                prices=input_ts,
                start_datetime=datetime_list[0],
                end_datetime=datetime_list[-1],
                pred_end_datetime=output_ts[-1],
                granularity=granularity_string,
                prediction_length=len(output_ts),
                mode=args.mode,
            )
        elif args.indicator == "bb":
            output_ts = sample["out_upper_bb"]
            designed_prompt = finance_bb_metaprompt_generation(
                text=text,
                prices=input_ts,
                start_datetime=datetime_list[0],
                end_datetime=datetime_list[-1],
                pred_end_datetime=output_ts[-1],
                granularity=granularity_string,
                prediction_length=len(output_ts),
                mode=args.mode,
            )
        elif args.indicator == "time":
            output_ts = sample["output_window"]
            designed_prompt = finance_mse_metaprompt_generation(
                text=text,
                prices=input_ts,
                start_datetime=datetime_list[0],
                end_datetime=datetime_list[-1],
                pred_end_datetime=output_ts[-1],
                granularity=granularity_string,
                prediction_length=len(output_ts),
                mode=args.mode,
            )   

        answer = model.inference(designed_prompt)
        answer = answer.strip().replace('"', '')

        predict_ts = parse_val_prediction_response(answer)
        predict_ts_orig = predict_ts
        predict_ts = np.interp( # type: ignore
            np.linspace(0, 1, len(output_ts)), 
            np.linspace(0, 1, len(predict_ts)), 
            predict_ts
        )
        
        res = {
            "filename": sample["filename"],
            "response": answer,
            "ground_truth": output_ts,
            "predict": predict_ts.tolist(),
        }
        result_list.append(res)
        
        save_to_json(res, details_path / sample["filename"])
        
        if args.indicator == "macd":
            first_half = sample["in_macd"]
        elif args.indicator == "bb":
            first_half = sample["in_upper_bb"]
        elif args.indicator == "time":
            first_half = sample["input_window"]
        plot_series(sample["filename"], first_half, output_ts, predict_ts_orig, visualizations_path)
        
        mse = np.mean((np.array(output_ts) - np.array(predict_ts)) ** 2)
        mae = np.mean(np.abs(np.array(output_ts) - np.array(predict_ts)))
        rmse = np.sqrt(mse)
        mape = calculate_mape(output_ts, predict_ts)
        
        if args.indicator == "macd" and mse > 10:
            print(f"{sample['filename']} failed mse", mse)
            epoch_results.append({
                "filename": sample["filename"],
                "failed": True,
                "epoch": idx + 1,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            })
            continue
        
        if args.indicator == "time" and mse > 100:
            print(f"{sample['filename']} failed mse ", mse)
            epoch_results.append({
                "filename": sample["filename"],
                "failed": True,
                "epoch": idx + 1,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            })
            continue

        if args.indicator == "bb" and mse > 100:
            print(f"{sample['filename']} failed mse ", mse)
            epoch_results.append({
                "filename": sample["filename"],
                "failed": True,
                "epoch": idx + 1,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            })
            continue    
        
        cumulative_mse.append(mse)
        cumulative_mae.append(mae)
        cumulative_rmse.append(rmse)
        cumulative_mape.append(mape)
        
        epoch_results.append({
            "filename": sample["filename"],
            "epoch": idx + 1,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mean_mse": np.mean(cumulative_mse),
            "mean_mae": np.mean(cumulative_mae),
            "mean_rmse": np.mean(cumulative_rmse),
            "mean_mape": np.mean(cumulative_mape),
        })
        save_to_json(epoch_results, f"{save_path}/epoch_results.json")
        print(
            "{}/{}: mse: {:.4f}, mae: {:.4f}, rmse: {:.4f}".format(
                idx, tot_samples, mse, mae, rmse
            )
        )
    except Exception as e:
        print(f"Skipping {idx} due to error: {e}")


summary = {
    "total_samples": len(result_list),
    "mse": np.mean(cumulative_mse),
    "mae": np.mean(cumulative_mae),
    "rmse": np.mean(cumulative_rmse),
    "mape": np.mean(cumulative_mape),
}

save_to_json(summary, f"{save_path}/final_results.json")
print(f"Processing complete. Results saved to {save_path}/final_results.json")