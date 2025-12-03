import torch
from tqdm import tqdm

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
    dataloader, device, vision_model, text_model, preprocess, tokenizer, all, args=None,
):
    if all:
        all_image_features = []
        all_text_features = []
    else:
        all_image_features = [torch.zeros(1), torch.zeros(1)]
        all_text_features = torch.empty(3,3)
    pooler_image_features = []
    all_labels = []
    
    pooler_text_features = []
    all_ids = []
    with torch.no_grad():
        for images, texts, labels, ids in tqdm(dataloader):
            # texts = clip.tokenize(texts,truncate=True)
            # images = preprocess(images, return_tensors="pt")
            #print(type(images), type(texts), type(labels), type(ids))
            #print(texts)
            texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            features = vision_model(**images)
            text_features = text_model(**texts.to(device))
            if all:
                all_image_features.append(features.last_hidden_state.detach().cpu())
                all_text_features.append(text_features.last_hidden_state.detach().cpu())
            pooler_image_features.append(features.pooler_output.detach().cpu())
            
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