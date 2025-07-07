import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
# sys.path.append("../..")
import os
from tqdm import tqdm
import pandas as pd

from models.model_factory import ModelFactory
from meta_prompt import finance_classification_metaprompt_generation, parse_cls_response
from utils import save_to_json, calculate_acc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="path to the dataset")
parser.add_argument("--save_path", type=str, help="path to save the results")
parser.add_argument("--model_type",  type=str, help="deepseek or llama")
parser.add_argument("--model",  type=str, help="model name")
parser.add_argument(
    "--mode",
    type=str,
    default="combined",
    help="choose from timeseries_only, text_only, combined",
)
args = parser.parse_args()

data_list = []
df = pd.read_parquet(args.dataset_path)
filename = Path(args.dataset_path).name
index = int(filename.split('_')[0]) if '_' in filename else 0

for _, row in df.iterrows():
    trend = row.get("trend", {})
    text = row.get("text", {})

    if not isinstance(trend.get("output_bin_label"), str):
        continue

    extracted_data = {
        "filename": filename,
        "index": index,
        "input_timestamps": row.get("input_timestamps"),
        "input_window": row.get("input_window"),
        "output_timestamps": row.get("output_timestamps"),
        "output_window": row.get("output_window"),
        "percentage_change": trend.get("output_percentage_change"),
        "bin_label": trend.get("output_bin_label"),
        "text": text.get("content"),
        "timestamp_ms": datetime.utcfromtimestamp(text.get("timestamp_ms", 0) / 1000)
            if isinstance(text, dict) and "timestamp_ms" in text else None
    }
    data_list.append(extracted_data)

os.makedirs(Path(args.save_path), exist_ok=True)

model = ModelFactory.get_model(model_type=args.model_type, model_name=args.model)

result_list = []
tot_samples = len(data_list)
print("Evaluating {} samples......".format(tot_samples))

for idx, sample in tqdm(enumerate(data_list), total=tot_samples):
    datetime_list = [
        datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S') for s in sample['input_timestamps']
    ]
    
    text = sample['text']
    prices = sample['input_window']
    
    designed_prompt = finance_classification_metaprompt_generation(
        text=text,
        timestamps=datetime_list,
        prices=prices,
        mode=args.mode
    )
    try:
        answer = model.inference(designed_prompt)
        answer = answer.strip().replace('"', '')
        gt = sample['bin_label']
        predict = parse_cls_response(answer)
        res = {
            "cnt": len(result_list),
            "filename": sample["filename"], 
            "ground_truth": gt, 
            "predict": predict,
            "answer": answer
        }
        result_list.append(res)
        acc_5way = calculate_acc(result_list)
        acc_3way = calculate_acc(
            result_list, 
            regrouped_labels={
                "<-4%": 'negtive',
                "-2% ~ -4%": 'negtive',
                "-2% ~ +2%": 'neutral',
                "+2% ~ +4%": 'positive',
                ">+4%": 'positive'
                }
            )
        result_list[-1]["accumulated_acc_5way"] = acc_5way
        result_list[-1]["accumulated_acc_3way"] = acc_3way
        print("{}/{}: ground_truth: {}; predicted: {}.".format(idx, tot_samples, gt, predict))
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if (idx +1) % 20 == 0:
        save_to_json(result_list, save_path=f"{args.save_path}/results.json")


final_acc_5way = result_list[-1]["accumulated_acc_5way"]
final_acc_3way = result_list[-1]["accumulated_acc_3way"]
print("Final results: 5-way-acc {:.4f}%, 3-way-acc {:.4f}%".format(final_acc_5way*100, final_acc_3way*100))


save_to_json(result_list, save_path=f"{args.save_path}/results.json")
print(f"Processing complete. Results saved to {args.save_path}/results.json")