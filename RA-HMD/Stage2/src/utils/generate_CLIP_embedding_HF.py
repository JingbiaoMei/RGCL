import argparse
import torch
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel

import sys
import os

sys.path.append('./src')
from data_loader.dataset import (
    get_Dataloader,
)
from extract_CLIP_features import extract_clip_features_HF

device = "cuda" if torch.cuda.is_available() else "cpu"

# This script generates CLIP CLS embeddings and the last hidden state of the model,
# Last hidden state represents the token embedding for the texts and the patch embedding for the images
# Here we use huggingface CLIP model rather than the OpenAI CLIP model


def parse_args_sys(args_list=None):
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/CLIP_Embedding",
        help="The path to save results.",
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="The clip model to use",
    )
    arg_parser.add_argument(
        "--image_size", type=int, default=336, help="The image size to use")
    arg_parser.add_argument("--dataset", type=str, default="FB", help="FB or MMHS")
    # ===== Inference Configuration ===== #
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--all", type=bool, default=False)
    args = arg_parser.parse_args()
    return args


def main(args):
    if os.path.exists("{}/{}".format(args.EXP_FOLDER, args.dataset)) == False:
        os.makedirs("{}/{}".format(args.EXP_FOLDER, args.dataset))
    

    # Load the CLIP model
    Vision_model = CLIPVisionModel.from_pretrained(args.model)
    Text_model = CLIPTextModel.from_pretrained(args.model)
    preprocess = CLIPProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    if device == "cuda":
        Vision_model.cuda().eval()
        Text_model.cuda().eval()
    else:
        Vision_model.eval()
        Text_model.eval()

    # Initialise dataset for FB
    if args.dataset == "FB":
        train, dev_seen, test_seen, test_unseen = get_Dataloader(
            preprocess,
            batch_size=args.batch_size,
            num_workers=24,
            train_batch_size=args.batch_size,
            image_size=args.image_size,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen, test_unseen]
        name_list = ["train", "dev_seen", "test_seen", "test_unseen"]
    elif args.dataset == "HarMeme" or args.dataset == "MMHS" or args.dataset == "Propaganda" or args.dataset == "Tamil" or args.dataset == "HarmC" or args.dataset == "HarmP" or args.dataset == "MultiOFF":
        train, dev_seen, test_seen = get_Dataloader(
            preprocess,
            batch_size=args.batch_size,
            num_workers=24,
            train_batch_size=args.batch_size,
            image_size=args.image_size,
            dataset=args.dataset,
        )
        loader_list = [test_seen, train, dev_seen ]
        name_list = ["test_seen", "train", "dev_seen"]
    elif "Memotion" in args.dataset:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess,
            batch_size=args.batch_size,
            num_workers=24,
            train_batch_size=args.batch_size,
            image_size=args.image_size,
            dataset=args.dataset,
        )
        loader_list = [ train, dev_seen, test_seen ]
        name_list = ["train", "dev_seen", "test_seen"]
        
    else:
        raise ValueError("Dataset not supported")
    # Get CLIP features and ground truth labels

    # Get the image ids in the same order as the features

    # Save it to a dictionary and save it as a .pt file
    
    for loader, name in zip(
        loader_list,
        name_list,
    ):
        (
            all_img_feats,
            all_text_feats,
            pooler_img_feats,
            pooler_text_feats,
            labels,
            ids,
        ) = extract_clip_features_HF(loader, device, Vision_model, Text_model, preprocess, tokenizer, args.all)
        torch.save(
            {
                "ids": ids,
                "img_feats": pooler_img_feats,
                "text_feats": pooler_text_feats,
                "labels": labels,
            },
            "{}/{}/{}_{}_HF.pt".format(
                args.EXP_FOLDER, args.dataset, name, str(args.model).replace("/", "_")
            ),
        )
        if args.all:
            torch.save(
                {
                    "ids": ids,
                    "img_feats": all_img_feats,
                    "text_feats": all_text_feats,
                    "labels": labels,
                },
                "{}/{}/{}_{}_HF_All.pt".format(
                    args.EXP_FOLDER, args.dataset, name, str(args.model).replace("/", "_")
                ),
            )


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
