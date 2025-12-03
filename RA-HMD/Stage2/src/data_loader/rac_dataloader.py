from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm



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


"""def modality_fusion(img_feats, text_feats, fusion_mode="concat", hf=False, normalize=False):
    if normalize:
        img_feats = torch.nn.functional.normalize(img_feats, p=2, dim=1)
        text_feats = torch.nn.functional.normalize(text_feats, p=2, dim=1)

    if fusion_mode == "separate":
        # Return 3D tensor with shape (len_dataset, 2, feat_dim)
        img_feats = img_feats.unsqueeze(1)
        text_feats = text_feats.unsqueeze(1)
        return torch.cat((img_feats, text_feats), dim=1)

    if fusion_mode == "concat":
        if hf:
            feature = torch.cat((img_feats, text_feats), dim=0)
        else:
            # TODO
            # COLBERT implementation
            feature = torch.cat((img_feats, text_feats), dim=1)
    elif fusion_mode == "dot":
        # TODO
        # HateCLIPPER paper
        feature = img_feats * text_feats
    elif fusion_mode == "align":
        feature = torch.mul(img_feats, text_feats)
    elif fusion_mode == "cross":
        feature = torch.bmm(img_feats.unsqueeze(
            2), text_feats.unsqueeze(1)).flatten(1, 2)
    print(feature.shape)
    return feature"""


def CLIP2Dataloader(*datasets, batch_size=128, return_dataset=False, normalize=False):
    """

    Args:
        *datasets (list): list of datasets, make sure train dataset is the first one
        batch_size (int, optional): batch size. Defaults to 128.
        fusion_mode (str, optional): Multimodality fusion mode. Defaults to "concat".
        hf (bool, optional): using huggingface CLIP features or not. Defaults to False.

    Returns:
        list: list of data loaders
    """

    dataloader_list = []

    dataset_list = []
    for index, dataset in enumerate(datasets):

        ids,  img_feats, text_feats, labels = dataset
        # Modality fusion is something we might want to move to modelling to save memory
        # For larger dataset, this causes OOM
        # Solved, for hate-clipper, we do modaility fusion in the model by passing
        # argument "separate" to parameter fusion_mode, the feats will be a
        # 3D tensor with shape (batch_size, 2, feat_dim)
        # feats = modality_fusion(img_feats.cpu().float(), text_feats.cpu().float(), fusion_mode=fusion_mode, hf=hf)

        feats = (img_feats.float(), text_feats.float())

        # change datatype to float32
        # feats = feats.float()
        dataset = RACDataset(feats, ids, labels)
        if return_dataset:
            dataset_list.append(dataset)
        if index == 0:
            # For training set, shuffle the data
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            # For validation set, do not shuffle the data
            # Batch size is 4 times larger than training set
            dataloader = DataLoader(
                dataset, batch_size=batch_size*4, shuffle=False, num_workers=0)
        dataloader_list.append(dataloader)
    if return_dataset:
        return dataloader_list, dataset_list
    else:
        return dataloader_list
# Dataset for Retrieval augmented classification


class RACDataset(Dataset):
    def __init__(self, feats, ids, labels):
        self.image_feats = feats[0]
        self.text_feats = feats[1]

        self.ids = ids
        self.labels = labels

    def __getitem__(self, index):
        return {"ids": self.ids[index], "image_feats": self.image_feats[index], "text_feats": self.text_feats[index], "labels": self.labels[index]}

    def __len__(self):
        return len(self.ids)


def get_Dataloader_FB(img_feats, text_feats, ids, labels, batch_size):
    dataset = RACDataset(img_feats, text_feats, ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    return dataloader


def LMM2Dataloader(*datasets, batch_size=128,):
    dataloader_list = []
    for index, dataset in enumerate(datasets):
        ids, feats, labels = dataset
        dataset = RACLMMDataset(feats, ids, labels)
        if index == 0:
            # For training set, shuffle the data
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            dataloader = DataLoader(
                dataset, batch_size=batch_size*4, shuffle=False, num_workers=0)
        dataloader_list.append(dataloader)
    return dataloader_list
    
class RACLMMDataset(Dataset):
    def __init__(self, feats, ids, labels):
        self.feats = feats

        self.ids = ids
        self.labels = labels
        
    def __getitem__(self, index):
        return {"ids": self.ids[index], "feats": self.feats[index], "labels": self.labels[index]}

    def __len__(self):
        return len(self.ids)