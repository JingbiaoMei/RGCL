from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from tqdm import tqdm
import numpy as np
from data_loader.feature_loader import get_attrobj_from_ids
from sklearn.preprocessing import MultiLabelBinarizer
import re
"""
For linear probe/fine tune/zero-shot
"""


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (str)):
        return data
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data.to(device)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


## Feature loader for loading pre-extracted CLIP features
# Used for fine_tune/zero_shot/linear_classifier
class feature_dataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = features.clone().detach().type(torch.float32)
        self.labels = labels.clone().detach().type(torch.float32).to(device)
        self.length = self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return self.length


# Image and Text Dataset
# Used for fine_tune/zero_shot/linear_classifier/generate clip embeddings
class image_text_dataset(Dataset):
    def __init__(
        self,
        img_data,
        preprocess,
        device="cuda",
        image_size=224,
    ):
        # img_data is a list of [list_image_path,list_text,list_label,list_ids]
        list_image_path, list_text, list_label, list_ids = img_data
        self.image_path = list_image_path
        self.text = list_text  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.label = list_label
        self.list_ids = list_ids
        self.preprocess = preprocess
        self.device = device
        # HuggingFace Tokenizer has different definition vs OPENAI CLIP
        self.image_size = image_size

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        image = self.preprocess(
            images=Image.open(self.image_path[idx]).convert('RGB').resize((self.image_size, self.image_size)), return_tensors="pt"
        )
        image["pixel_values"] = image["pixel_values"].squeeze()

        text = self.text[idx]
  
        label = self.label[idx]
        label = torch.tensor(label)
        return image, text, label, self.list_ids[idx] 

def get_values_from_gt(dataset, split):
    """
    Extract image path, text, gt_label and image ids from gt files
    input: dataset name, split name
    output: list of ... for input to image_text_dataset
    """
    # Read the ground truth file
    if dataset != "MultiOFF" and "Memotion" not in dataset:
        gt_file = "./data/gt/" + dataset + "/" + split + ".jsonl"
        gt_df = pd.read_json(gt_file, lines=True, dtype=False)

        # Get the ordered list of image ids
        list_ids = gt_df["id"].values
        # Get the ordered list of text and labels
        list_text = gt_df["text"].to_list()
    
    else:
        gt_df = None
    # Get the ordered list of image paths
    list_image_path = []

    
    
    if dataset == "FB" or dataset.lower() == "fb" or dataset == "PrideMM":
        list_label = gt_df["label"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".png")
    elif dataset == "MAMI":
        list_label = gt_df["label"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".jpg")
    elif dataset.lower() == "harmp" or dataset.lower() == "harmeme":

        list_label = gt_df["labels"]
        list_label_converted = []
        for item in list_label:
            if 'not harmful' in item:
                list_label_converted.append(0)
            else:
                list_label_converted.append(1)  # harmful
        assert len(list_label_converted) == len(list_label)
        list_label = list_label_converted
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".png")
    elif dataset == "HarmC" :
        # The HarmC is the same as HarMeme, but here we refer to 3 classes
        # 0: not harmful, 1: 
        list_label = gt_df["labels"]
        list_label_converted = []
        for item in list_label:
            if 'not harmful' in item:
                list_label_converted.append(0)
            elif "somewhat harmful" in item:
                list_label_converted.append(1)
            elif "very harmful" in item:
                list_label_converted.append(2)
        assert len(list_label_converted) == len(list_label)
        list_label = list_label_converted
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id + ".png")
    elif dataset == "Propaganda":
        #list_label = gt_df["label"].to_list()
        list_image = gt_df["image"].to_list()
        for image_id in list_image:
            list_image_path.append("./data/image/" + dataset + "/All/" + image_id)
        fine_grained_labels = ['Black-and-white Fallacy/Dictatorship', 'Name calling/Labeling', 'Smears', 'Reductio ad hitlerum', 'Transfer', 'Appeal to fear/prejudice', \
            'Loaded Language', 'Slogans', 'Causal Oversimplification', 'Glittering generalities (Virtue)', 'Flag-waving', "Misrepresentation of Someone's Position (Straw Man)", \
            'Exaggeration/Minimisation', 'Repetition', 'Appeal to (Strong) Emotions', 'Doubt', 'Obfuscation, Intentional vagueness, Confusion', 'Whataboutism', 'Thought-terminating cliché', \
            'Presenting Irrelevant Data (Red Herring)', 'Appeal to authority', 'Bandwagon']
        mlb = MultiLabelBinarizer().fit([fine_grained_labels])
        gt_df = gt_df.join(pd.DataFrame(mlb.transform(gt_df['labels']), 
                                        columns=mlb.classes_, 
                                        index=gt_df.index))
        list_label = []
        for i in range(len(list_ids)):
            list_label.append(gt_df.iloc[i][fine_grained_labels].values.tolist())
            # A list of list 
            # Each sublist is something like [1,1,1,0,...,0] with 22 elements
            
    elif dataset == "Tamil":
        list_label = gt_df["label"].to_list()
        list_image = gt_df["image_id"].to_list()
        for img_id in list_image:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id)
    elif dataset == "MMHS":
        for index, text in enumerate(list_text):
            # Remove the url and @user
            
            text = re.sub(r' https\S+', "", text)
            text = re.sub(r'@\S+ ', "", text)
            list_text[index] = text
        
        list_label = gt_df["label"].to_list()
        for img_id in list_ids:
            list_image_path.append("./data/image/" + dataset + "/All/" + str(img_id) + ".jpg")
    elif dataset == "MultiOFF":
        if split == "train":
            gt_file = "./data/gt/" + dataset + "/" + "Training_meme_dataset.csv"
        elif split == "val":
            gt_file = "./data/gt/" + dataset + "/" + "Validation_meme_dataset.csv"
        elif split == "test":
            gt_file = "./data/gt/" + dataset + "/" + "Testing_meme_dataset.csv"
        gt_df = pd.read_csv(gt_file)
        list_image = gt_df["image_name"].to_list()
        list_ids = list_image
        list_text = gt_df["sentence"].to_list()
        list_label_text = gt_df["label"].to_list()
        for img_id in list_image:
            list_image_path.append("./data/image/" + dataset + "/All/" + img_id)
        list_label = []
        for label in list_label_text:
            if label == "Non-offensiv":
                list_label.append(0)
            elif label == "offensive":
                list_label.append(1)
            else:
                print("MultiOFF: Error, do not know the label")        
    elif "Memotion" in dataset:
        # Humour, Sarcasm, Offense, Motivation
        if split == "train":
            gt_file = "./data/gt/" + "Memotion" + "/" + "labels.csv"
            gt_df = pd.read_csv(gt_file)
            list_image = gt_df["image_name"].to_list()
            for img_id in list_image:
                list_image_path.append("./data/image/" + "Memotion" + "/All/" + img_id)
            #print("start to test images")
            #for i, image_pth in tqdm(enumerate(list_image_path)):
            #    try:
            #        Image.open(image_pth).convert('RGB')
            #    except:
            #        print("Error found in image {}".format(i))
            list_ids = list_image
            list_text = gt_df["text_corrected"].to_list()
            list_text_supplement = gt_df["text_ocr"].to_list()
            #print(list_text[119])
            #print(list_text[119] == "nan")
            #print(list_text[119] == list_text[119])
            for i, (text, text_sup) in enumerate(zip(list_text, list_text_supplement)):
                # Address nan in input text
                if text != text:
                    print("{} Text corrected is empty, replace with OCR".format(i))
                    if text_sup == text_sup:
                        list_text[i] = text_sup
                    else:
                        # sine text sup is also nan, using empty string 
                        list_text[i] = " "
            
            list_label = []
            if dataset == "Memotion_H":
                humour = gt_df["humour"].to_list()
                for item in humour:
                    if item == "not_funny":
                        list_label.append(0)
                    else:
                        list_label.append(1)
   
            elif dataset == "Memotion_S":
                sarcasm = gt_df["sarcasm"].to_list()
                for item in sarcasm:
                    if item == "not_sarcastic":
                        list_label.append(0)
                    else:
                        list_label.append(1)
            
            elif dataset == "Memotion_O":
                offensive = gt_df["offensive"].to_list()
                for item in offensive:
                    if item == "not_offensive":
                        list_label.append(0)
                    else:
                        list_label.append(1)
            elif dataset == "Memotion_M":
                motivation = gt_df["motivational"].to_list()
                for item in motivation:
                    if item == "not_motivational":
                        list_label.append(0)
                    else:
                        list_label.append(1)
            else:
                print("Memotion: Error, do not know the task within this dataset")
        else:
            gt_file_1 = "./data/gt/" + "Memotion" + "/" + "2000_testdata.csv"
            gt_file_2 = "./data/gt/" + "Memotion" + "/" + "Meme_groundTruth.csv"
            gt_df = pd.read_csv(gt_file_1)
            gt_df_labels = pd.read_csv(gt_file_2)
            list_image = gt_df["Image_name"].to_list()
            for img_id in list_image:
                list_image_path.append("./data/image/" + "Memotion" + "/All/" + img_id)
            #print("start to test images")
            #for i, image_pth in tqdm(enumerate(list_image_path)):
            #    try:
            #        Image.open(image_pth).convert('RGB')
            #    except:
            #        print("Error found in image {}".format(i))
            
            list_ids = list_image
            list_text = gt_df["corrected_text"].to_list()
            list_text_supplement = gt_df["OCR_extracted_text"].to_list()
            for i, (text, text_sup) in enumerate(zip(list_text, list_text_supplement)):
                if text != text:
                    print("{} Text corrected is empty, replace with OCR".format(i))
                    if text_sup == text_sup:
                        list_text[i] = text_sup
                    else:
                        # sine text sup is also nan, using empty string 
                        list_text[i] = " "
            labels_pool = gt_df_labels["Labels"].to_list()
            labels_pool = [ label.split("_")[1] for label in labels_pool]
            list_label = []
            if dataset == "Memotion_H":
                for label in labels_pool:
                    #print(label)
                    list_label.append(int(label[0]))
            elif dataset == "Memotion_S":
                for label in labels_pool:
                    list_label.append(int(label[1]))
            elif dataset == "Memotion_O":
                for label in labels_pool:
                    list_label.append(int(label[2]))
            elif dataset == "Memotion_M":
                for label in labels_pool:
                    list_label.append(int(label[3]))
            else:
                print("Memotion: Error, do not know the task within this dataset")
                    
    else:
        raise ValueError("{} Dataset not supported".format(dataset))
    return list_image_path, list_text, list_label, list_ids


def get_img_ids(dataset, split):
    """
    get ordered image ids from gt files
    """
    gt_file = "./data/gt/" + dataset + "/" + split + ".jsonl"
    gt_df = pd.read_json(gt_file, lines=True, dtype=False)
    list_ids = gt_df["id"].values
    return list_ids


# Load values into DL
def get_Dataloader(
    preprocess,
    batch_size=128,
    num_workers=4,
    train_batch_size=32,
    device="cuda",
    image_size=224,
    dataset="FB",
):
    imgtxt_dataset = image_text_dataset(
        get_values_from_gt(dataset, "train"),
        preprocess,
    )
    train = DataLoader(
        imgtxt_dataset, batch_size=train_batch_size, num_workers=num_workers
    )  # Define your own dataloader
    train = DeviceDataLoader(train, device)
    imgtxt_dataset = image_text_dataset(
        get_values_from_gt(dataset, "dev_seen" if dataset == "FB" else "val"),
        preprocess,
        image_size=image_size,
    )
    dev_seen = DataLoader(
        imgtxt_dataset, batch_size=batch_size, num_workers=num_workers
    )  # Define your own dataloader
    dev_seen = DeviceDataLoader(dev_seen, device)
    imgtxt_dataset = image_text_dataset(
        get_values_from_gt(dataset, "test_seen" if dataset == "FB" else "test"),
        preprocess,
        image_size=image_size,
    )
    test_seen = DataLoader(
        imgtxt_dataset, batch_size=batch_size, num_workers=num_workers
    )  # Define your own dataloader
    test_seen = DeviceDataLoader(test_seen, device)
    
    if dataset == "FB":
        imgtxt_dataset = image_text_dataset(
            get_values_from_gt(dataset, "test_unseen"),
            preprocess,
            image_size=image_size,
        )
        
        test_unseen = DataLoader(
            imgtxt_dataset, batch_size=batch_size, num_workers=num_workers
        )  # Define your own dataloader
        test_unseen = DeviceDataLoader(test_unseen, device)
        return train, dev_seen, test_seen, test_unseen
    else:
        return train, dev_seen, test_seen





# This function extract the CLS token from the CLIP model with CLIP python package
def extract_clip_features(dataloader, device, model):
    CLS_image_features = []
    all_labels = []
    CLS_text_features = []
    all_ids = []
    with torch.no_grad():
        for images, texts, labels, ids in tqdm(dataloader):
            # texts = clip.tokenize(texts,truncate=True)
            features = model.encode_image(images.to(device))

            text_features = model.encode_text(texts.to(device))
            CLS_image_features.append(features)
            CLS_text_features.append(text_features)
            all_labels.append(labels)
            all_ids.append(ids)

    return (
        torch.cat(CLS_image_features),
        torch.cat(CLS_text_features),
        torch.cat(all_labels),
        all_ids,
    )


# This function extract both the last hidden state and the CLS token
# from the CLIP model with HuggingFace transformers package
# last hidden state i.e., the token embedding and the patch embedding
def extract_clip_features_HF(
    dataloader, device, vision_model, text_model, preprocess, tokenizer
):
    all_image_features = []
    pooler_image_features = []
    all_labels = []
    all_text_features = []
    pooler_text_features = []
    all_ids = []
    with torch.no_grad():
        for images, texts, labels, ids in tqdm(dataloader):
            # texts = clip.tokenize(texts,truncate=True)
            # images = preprocess(images, return_tensors="pt")
            texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            features = vision_model(**images)
            text_features = text_model(**texts.to(device))
            all_image_features.append(features.last_hidden_state.detach().cpu())
            pooler_image_features.append(features.pooler_output.detach().cpu())
            all_text_features.append(text_features.last_hidden_state.detach().cpu())
            pooler_text_features.append(text_features.pooler_output.detach().cpu())
            all_labels.append(labels)
            all_ids.append(ids)

    return (
        torch.cat(all_image_features),
        all_text_features,
        torch.cat(pooler_image_features),
        torch.cat(pooler_text_features),
        torch.cat(all_labels),
        all_ids,
    )


# def get_linear_prob_features(dataloader, device, model)


"""
For Retrieval based system
"""


def load_feats_from_CLIP(path, dataset, model, all=False):
    """
    return the pre-extracted features from CLIP model

    """
    if dataset == "FB":
        """
        load the features for FB dataset, which contains train, dev, test_seen, test_unseen sets
        each sets contains ids, img_feats, text_feats, labels
        """
        train, dev, test_seen, test_unseen = load_feats_FB(path, model)
        
        # All is for concatenating all the splits 
        # into one whole dataset
        if all:
            all = concate_all_splits_FB(train, dev, test_seen, test_unseen)
            return train, dev, test_seen, test_unseen, all
        else:
            return train, dev, test_seen, test_unseen
    elif dataset in {"HarMeme", "HarmP", "Propaganda", "Tamil", "MMHS", "MultiOFF"} or any(substring in dataset for substring in {"Memotion", "PrideMM", "MAMI"}):

        # Load features for general datasets with train, dev, and test_seen splits.
        train, dev, test_seen = load_feats_general(path, model, dataset)
        return train, dev, test_seen

    else:
        raise NotImplementedError(f"Dataset '{dataset}' is not supported.")


# Used for FB dataset to stats the whole dataset
# Not used for actual training
def concate_all_splits_FB(train, dev, test_seen, test_unseen):
    """
    This function takes all the splits and concate them into one whole dataset
    to test the number of unique images in the whole dataset
    """
    train_ids, train_img_feats, train_text_feats, train_labels = train
    dev_ids, dev_img_feats, dev_text_feats, dev_labels = dev
    (
        test_seen_ids,

        test_seen_img_feats,
        test_seen_text_feats,
        test_seen_labels,
    ) = test_seen
    (
        test_unseen_ids,
        test_unseen_img_feats,
        test_unseen_text_feats,
        test_unseen_labels,
    ) = test_unseen
    all_ids = train_ids + dev_ids + test_seen_ids + test_unseen_ids
    all_img_feats = torch.cat(
        (train_img_feats, dev_img_feats, test_seen_img_feats, test_unseen_img_feats),
        dim=0,
    )
    all_text_feats = torch.cat(
        (
            train_text_feats,
            dev_text_feats,
            test_seen_text_feats,
            test_unseen_text_feats,
        ),
        dim=0,
    )
    all_labels = torch.cat(
        (train_labels, dev_labels, test_seen_labels, test_unseen_labels), dim=0
    )
    return [all_ids,  all_img_feats, all_text_feats, all_labels]


def load_feats_split(path, dataset=None):
    """
    load features for FB dataset for each dataset splits
        which contains
        ids: the image ids in the same order as the features
        ids_dics: maps the image ids to the order of the image
        img_feats: the features extracted by CLIP model
        text_feats: the features extracted by CLIP model
        labels: ground truth labels
    The features are extracted and defined in generate_CLIP_embedding.py
    """
    dict = torch.load(path)
    ids = dict["ids"]
    # flatten the python 2d ids list (batch size as the row) to a 1d list
    ids = [item for sublist in ids for item in sublist]
    
        
    # ids map the order of the image to image ids
    # ids_dics maps the image ids to the order of the image
    #ids_dics = {k: v for v, k in enumerate(ids)}

    img_feats = dict["img_feats"]
    text_feats = dict["text_feats"]
    labels = dict["labels"]

    if dataset == "MMHS":
        for index, label in enumerate(labels):

            if label > 0:

                labels[index] = 1

    return [ids, img_feats, text_feats, labels]

def load_feats_FB(path, model):
    dataset = "FB"
    train = load_feats_split("{}/{}/train_{}.pt".format(path, dataset, model))
    dev = load_feats_split("{}/{}/dev_seen_{}.pt".format(path, dataset, model))
    test_seen = load_feats_split(
        "{}/{}/test_seen_{}.pt".format(path, dataset, model)
    )
    test_unseen = load_feats_split(
        "{}/{}/test_unseen_{}.pt".format(path, dataset, model)
    )
    return train, dev, test_seen, test_unseen

def load_feats_general(path, model, dataset):
    train = load_feats_split("{}/{}/train_{}.pt".format(path, dataset, model))
    dev = load_feats_split("{}/{}/dev_seen_{}.pt".format(path, dataset, model))
    test_seen = load_feats_split(
        "{}/{}/test_seen_{}.pt".format(path, dataset, model)
    )
    return train, dev, test_seen




def get_sparse_data_train_FB(
    img_feature_file,
    gt_train_file,
    attribute=True,
    objects_conf_threshold=None,
):
    """Organize the sparse data for training set of FB dataset
    Sparse data are the text based data opposed to the dense data from CLIP embeddings

    Args:
        img_feature_file (string to get list of dict for img_dict): the object detection results
                            from the images, contains bounding box predictions

        gt_train_file (string: file path): the ground truth file (json) for training set of FB dataset
                                to get the ground truth labels, captions, and order of the image ids

        attribute (bool, optional): Including attributes or not, Defaults to True.

        objects_conf_threshold (float, optional): The threshold for the confidence level for an detected object
                                        Defaults to None.

    Returns:
        dictionary of dictionary: sparse_retrieval_train
        sprase_retrieval_train["img_id"] = ["img_id", "text", "label", "objects", "attributes", "objects_conf"]
    """

    sparse_retrieval_train = {}
    gt_train = pd.read_json(gt_train_file, lines=True, dtype=False)
    gt_train.set_index("id", inplace=True)
    feature_path = "./data/features/" + img_feature_file + ".tsv"
    img_dict = get_attrobj_from_ids(feature_path)
    # iterate through the image dictionary
    for img_id in img_dict:
        # get the image id
        img_id = img_dict[img_id]["img_id"]

        # get the object names
        object_names = img_dict[img_id]["object_names"]
        # get the object confidences
        objects_conf = img_dict[img_id]["objects_conf"]

        # get the attribute names
        if attribute:
            attribute_names = img_dict[img_id]["attribute_names"]
        else:
            # if attribute is false, use empty list
            attribute_names = [""] * len(object_names)
        # get the attribute confidences
        # attrs_conf = img_dict[img_id]["attrs_conf"]

        if objects_conf_threshold:
            # Since the confidences are sorted, we can just take the first n
            num_objects = np.sum(objects_conf >= objects_conf_threshold)
            # If all the confidences are smaller than the threshold,
            # then we just use the first one
            if num_objects == 0:
                num_objects = 1
            object_names = object_names[:num_objects]
            attribute_names = attribute_names[:num_objects]

        # Concat the object and attribute names for each object
        attobject_list = [
            obj + " " + attr for obj, attr in zip(object_names, attribute_names)
        ]

        # get the ground truth captions and concat with the object and attribute names
        if img_id in gt_train.index:
            #
            sparse_retrieval_train[img_id] = {
                "img_id": img_id,
                "text": gt_train.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_train.loc[img_id]["label"],
                "objects": object_names,
                "attributes": attribute_names,
                "objects_conf": objects_conf,
            }
    # Now that we have the dictionary of the training data,
    # We need to rearrange the order of the data to match the order of the dataloader and json file
    # Since the order of the image ids from object detection is different

    # We can get the order of the image ids from the gt json file
    # and then rearrange the order of the dictionary
    
    
    
    return sparse_retrieval_train


def load_feats_from_LMM(path, dataset, all=False):
    """
    return the pre-extracted features from LMM models
    """
    train_path = path
    if dataset.lower() == "fb":
        val_path = path.replace("train", "dev_seen")
        test_path = path.replace("train", "test_seen")
        train = load_feats_split_LMM(train_path)
        val = load_feats_split_LMM(val_path)
        test = load_feats_split_LMM(test_path)
    else:
        val_path = path.replace("train", "val")
        test_path = path.replace("train", "test")
        train = load_feats_split_LMM(train_path, dataset)
        val = load_feats_split_LMM(val_path, dataset)
        test = load_feats_split_LMM(test_path, dataset)
    return train, val, test


def load_feats_split_LMM(path, dataset=None):
    """
    load features for FB dataset for each dataset splits
        which contains
        ids: the image ids in the same order as the features
        ids_dics: maps the image ids to the order of the image
        img_feats: the features extracted by CLIP model
        text_feats: the features extracted by CLIP model
        labels: ground truth labels
    The features are extracted and defined in generate_CLIP_embedding.py
    """
    dict = torch.load(path)
    ids = dict["ids"]
    # flatten the python 2d ids list (batch size as the row) to a 1d list
    # Check if the ids are in 2d or 1d
    if isinstance(ids[0], list):
        ids = [item for sublist in ids for item in sublist]
    
        
    # ids map the order of the image to image ids
    # ids_dics maps the image ids to the order of the image
    #ids_dics = {k: v for v, k in enumerate(ids)}

    feats = dict["feats"]

    labels = dict["labels"]
    
    # Convert from bfloat to float32 
    feats = feats.type(torch.float32)

    if dataset == "MMHS":
        for index, label in enumerate(labels):

            if label > 0:

                labels[index] = 1

    return [ids, feats, labels]