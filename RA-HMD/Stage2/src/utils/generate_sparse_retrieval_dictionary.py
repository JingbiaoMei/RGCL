import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import json
import pickle
import sys

sys.path.append('./src')
from utils.metrics import compute_metrics_retrieval_baseline
from utils.retrieval import retrieve_topk, get_sparse_data_FB, sparse_retrieval
from data_loader.feature_loader import get_attrobj_from_ids
import wandb
import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--feature_file",
        type=str,
        default="hm_attr3636_VinVLV1",
        help="The path to data.",
    )
    arg_parser.add_argument(
        "--output_path",
        type=str,
        default="./data/Sparse_Retrieval_Dict",
        help="The path to output.",
    )
    arg_parser.add_argument(
        "--dataset", type=str, default="FB", help="The dataset to use."
    )
    arg_parser.add_argument(
        "--use_attribute", default=True, type=lambda x: (str(x).lower() == 'true'),  help="Whether to use attribute."
    )
    arg_parser.add_argument(
        "--use_caption", default=False, type=lambda x: (str(x).lower() == 'true'),  help="Whether to use caption."
    )
    arg_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The confidence threshold for object detection bounding box to use.",
    )
    arg_parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="The additional name of the output file.",
    )
    arg_parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="The number of topk to retrieve.",
    )
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    feature_path = os.path.join(
        "./data/features/", args.dataset, args.feature_file + ".tsv"
    )
    args.output_path = os.path.join(args.output_path, args.dataset)
    use_attribute_name = "_attr" if args.use_attribute else ""
    use_caption_name = "_cap" if args.use_caption else ""
    
    args.output_name = "{}{}{}{}{}.json".format(
        args.feature_file, 
        use_attribute_name,
        use_caption_name,
        args.output_name,
        "_topk"+str(args.topk)
    )
    
    gt_dir = "./data/gt/"
    if args.dataset == "FB":
        gt_train_file = os.path.join(gt_dir, args.dataset, "train.jsonl")
        gt_train = pd.read_json(gt_train_file, lines=True, dtype=False)
        gt_val_file = os.path.join(gt_dir, args.dataset, "dev_seen.jsonl")
        gt_val = pd.read_json(gt_val_file, lines=True, dtype=False)
        gt_test_seen_file = os.path.join(gt_dir, args.dataset, "test_seen.jsonl")
        gt_test_seen = pd.read_json(gt_test_seen_file, lines=True, dtype=False)
        gt_test_unseen_file = os.path.join(gt_dir, args.dataset, "test_unseen.jsonl")
        gt_test_unseen = pd.read_json(gt_test_unseen_file, lines=True, dtype=False)
        
        img_dict = get_attrobj_from_ids(
            feature_path, 
            topk=-1,
            length=len(gt_train) + len(gt_val) + len(gt_test_seen) + len(gt_test_unseen)
        )
        (
            retrieve_train,
            retrieve_val,
            retrieve_test_seen,
            retrieve_test_unseen,
        ) = get_sparse_data_FB(
            img_dict,
            gt_train,
            gt_val,
            gt_test_seen,
            gt_test_unseen,
            attribute=args.use_attribute,
            objects_conf_threshold=args.threshold,
        )
    logging_dict = sparse_retrieval(
            retrieve_train, retrieve_train, retrieve_size=args.topk
        )
    
    # Dump to json
    with open(os.path.join(args.output_path, args.output_name), "w") as f:
        for k, v in logging_dict.items():
            
            json.dump(
                {
                    "id": k,
                    "label": int(retrieve_train[k]["label"]),
                    "retrieved_ids": v["retrieved_ids"],
                    "retrieved_labels": [int(value) for value in v["retrieved_label"]],
                },f
            )
            f.write("\n")

if __name__ == "__main__":
    main()