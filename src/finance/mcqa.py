"""
To run:

python src/finance/mcqa.py \
--dataset_folder /path/to/dataset \
--save_path /path/to/save/results \
--model_type deepseek or llama \
--model author/model HuggingFace ID\
--setting short or long
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
# sys.path.append("../..")
import os
from tqdm import tqdm

from models.model_factory import ModelFactory
from meta_prompt import finance_mcqa_metaprompt_generation
from utils import save_to_json, calculate_mcqa_acc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, help="path to the datasets")
parser.add_argument("--save_path", type=str, help="path to save the results")
parser.add_argument("--model_type",  type=str, help="deepseek or llama")
parser.add_argument("--model",  type=str, help="model name")
parser.add_argument("--setting",  type=str, help="short or long")
args = parser.parse_args()

data_list = []
directory_path = Path(args.dataset_folder)
for json_file in directory_path.glob("*.json"):
    with open(json_file, 'r') as file:
        data = json.load(file)
        sticker = json_file.name.split('_')[1].split('.')[0]
        extracted_data = {
            "filename": json_file.name,
            "sticker": sticker,
            "index": int(json_file.name.split('_')[0]),
            "input_timestamps": data.get("input_timestamps"),
            "input_window": data.get("input_window"),
            "output_timestamps": data.get("output_timestamps"),
            "output_window": data.get("output_window"),
            "question": data.get('MCQA').get('question'),
            "answer": data.get('MCQA').get('answer'),
            "text": data.get("text"),
            "published_utc": data.get("published_utc")
        }
        data_list.append(extracted_data)

os.makedirs(Path(args.save_path), exist_ok=True)

model = ModelFactory.get_model(model_type=args.model_type, model_name=args.model)

result_list = []
tot_samples = len(data_list)
print("Evaluating {} samples......".format(tot_samples))

for idx, sample in tqdm(enumerate(data_list), total=tot_samples):
    designed_prompt = finance_mcqa_metaprompt_generation(
        setting=args.setting,
        sticker=sample["sticker"],
        time1=datetime.fromtimestamp(sample["input_timestamps"][0]),
        time2=datetime.fromtimestamp(sample["input_timestamps"][-1]),
        in_price=sample["input_window"],
        news=sample["text"],
        time_news=sample["published_utc"],
        question=sample["question"]
    )
    try:
        answer = model.inference(designed_prompt)
        answer = answer.strip().replace('"', '')
        res = {
            "cnt": len(result_list),
            "filename": sample["filename"], 
            "question": sample["question"],
            "ground_truth": sample["answer"], 
            "predict": answer,
        }
        result_list.append(res)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    if (idx +1) % 20 == 0:
        save_to_json(result_list, save_path=f"{args.save_path}/results.json")


save_to_json(result_list, save_path=f"{args.save_path}/results.json")
accuracy = calculate_mcqa_acc(result_list)
print(f"Accuracy: {accuracy}%. Results saved to {args.save_path}/results.json")