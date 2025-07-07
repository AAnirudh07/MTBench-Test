"""
To run the code:

python src/finance/correlation_prediction.py \
--dataset_path /path/to/dataset.json \
--save_path /path/to/save/results \
--model_type deepseek or llama \
--model author/model \
--setting short or longs
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
# sys.path.append("../..")
import os
from tqdm import tqdm

from meta_prompt import finance_correlation_metaprompt_generation
from ..utils import save_to_json, calculate_correlation_acc
from models.model_factory import ModelFactory

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="path to the dataset")
parser.add_argument("--save_path", type=str, help="path to save the results")
parser.add_argument("--model_type",  type=str, help="deepseek or llama")
parser.add_argument("--model",  type=str, help="model name")
parser.add_argument("--setting",  type=str, help="short or long")
args = parser.parse_args()

data_list = []
with open(args.dataset_path) as file:
    data = json.load(file)
    sticker = args.dataset_path.name.split('_')[1].split('.')[0]
    extracted_data = {
        "filename": args.dataset_path.name,
        "sticker": sticker,
        "index": int(args.dataset_path.name.split('_')[0]),
        "input_timestamps": data.get("input_timestamps"),
        "input_window": data.get("input_window"),
        "output_timestamps": data.get("output_timestamps"),
        "output_window": data.get("output_window"),
        "correlation": data.get('news_price_correlation'),
        "text": data.get("text"),
        "published_utc": data.get("published_utc")
    }
    data_list.append(extracted_data)

os.makedirs(Path(args.save_path).parent, exist_ok=True)

model = ModelFactory.get_model(model_type=args.model_type, model_name=args.model)

result_list = []
tot_samples = len(data_list)
print("Evaluating {} samples......".format(tot_samples))

for idx, sample in tqdm(enumerate(data_list), total=tot_samples):
    designed_prompt = finance_correlation_metaprompt_generation(
        setting=args.setting,
        sticker=sample["sticker"],
        time1=datetime.fromtimestamp(sample["input_timestamps"][0]),
        time2=datetime.fromtimestamp(sample["input_timestamps"][-1]),
        in_price=sample["input_window"],
        news=sample["text"],
        time_news=sample["published_utc"]
    )
    try:
        answer = model.inference(designed_prompt)
        res = {
            "cnt": len(result_list),
            "filename": sample["filename"], 
            "ground_truth": sample["correlation"], 
            "predict": answer,
        }
        result_list.append(res)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if (idx +1) % 20 == 0:
        save_to_json(result_list, save_path=f"{args.save_path}/results.json")

 
save_to_json(result_list, save_path=f"{args.save_path}/results.json")
metric_results = calculate_correlation_acc(result_list)
metric_results["model"] = args.model
save_to_json(metric_results, save_path=f"{args.save_path}/final_results.json")
print(f"Processing complete. Results saved to {args.save_path}/final_results.json")