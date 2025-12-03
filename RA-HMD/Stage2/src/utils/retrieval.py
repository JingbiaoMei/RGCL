import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np
from easydict import EasyDict
from easydict import EasyDict
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# This function is implemented for the baseline experiment of DPR dpr_baseline.py
# Given the dense vectors of the image and text,
# we retrieve the top k image and text MM pairs
def retrieve_topk(
    database, query, alpha=1.0, beta=1.0, largest_retrieval=100, threshold=0.2
):
    """
    This function retrieve the top k image and text MM pairs for val/test
    with the given dense vectors

    input:
    database: the database set
    query: the query set
    alpha: the weight for image retrieval
    beta: the weight for text retrieval
    largest_retrieval: the largest number of retrieval neighbours
    threshold: the threshold for the similarity score

    The retrieved neighbours has to be larger than the threshold, if more than
    #largest_retrieval neighbours are larger than the threshold, then only the first
    #largest_retrieval neighbours are returned

    """

    ids,  img_feats, text_feats, labels = database
    q_ids, q_img_feats, q_text_feats, q_labels = query

    # Normalize the features
    img_feats_norm = img_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(img_feats_norm)
    text_feats_norm = text_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(text_feats_norm)

    q_img_feats_norm = q_img_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_img_feats_norm)
    q_text_feats_norm = q_text_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_text_feats_norm)
    feats = np.concatenate(
        (alpha * img_feats_norm, beta * text_feats_norm), axis=1)
    q_feats = np.concatenate(
        (alpha * q_img_feats_norm, beta * q_text_feats_norm), axis=1
    )

    # Get the dimension of the features
    dim = feats.shape[1]
    # Initialize the index
    index = faiss.IndexFlatL2(dim)
    index.add(feats)
    D, I = index.search(q_feats, largest_retrieval)

    logging_dict = EasyDict()

    for i, row in enumerate(D):
        # a list to record the ids of the retrieved example
        retrieved_ids = []
        # a list to record the similarity scores of the retrieved example
        retrieved_scores = []
        # a list to record the retrieved example's label
        retrieved_label = []
        for j, value in enumerate(row):
            # You have to retrieve at least one, no matter what the similarity score is
            if j == 0:
                retrieved_ids.append(ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(labels[I[i, j]].item())
            # if image is similar
            else:
                if value < threshold:
                    # for the temp list, we use the image ids rather than the ordered number
                    retrieved_ids.append(ids[I[i, j]])
                    retrieved_scores.append(value)
                    retrieved_label.append(labels[I[i, j]].item())
                # if larger than threshold,
                # then we can break the inside loop,
                # since the rest of the values are larger than the threshold
                else:
                    break
        # Record the number of images retrieved for each query
        no_retrieved = len(retrieved_ids)

        logging_dict[q_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }

    return logging_dict


# Structure the sparse retrieval data
def get_sparse_data_FB(
    img_dict,
    gt_train,
    gt_dev,
    gt_test_seen,
    gt_test_unseen,
    attribute=True,
    objects_conf_threshold=None,
):
    retrieve_train = {}
    retrieve_val = {}
    retrieve_test_seen = {}
    retrieve_test_unseen = {}
    gt_train.set_index("id", inplace=True)
    gt_dev.set_index("id", inplace=True)
    gt_test_seen.set_index("id", inplace=True)
    gt_test_unseen.set_index("id", inplace=True)

    # iterate through the image dictionary
    for img_id in img_dict:
        # get the image id
        # img_id = img_dict[img_id]["img_id"]

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
            retrieve_train[img_id] = {
                "text": gt_train.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_train.loc[img_id]["label"],
            }
        elif img_id in gt_dev.index:
            retrieve_val[img_id] = {
                "text": gt_dev.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_dev.loc[img_id]["label"],
            }
        elif img_id in gt_test_seen.index:
            retrieve_test_seen[img_id] = {
                "text": gt_test_seen.loc[img_id]["text"]
                + " "
                + " ".join(attobject_list),
                "label": gt_test_seen.loc[img_id]["label"],
            }
        elif img_id in gt_test_unseen.index:
            retrieve_test_unseen[img_id] = {
                "text": gt_test_unseen.loc[img_id]["text"]
                + " "
                + " ".join(attobject_list),
                "label": gt_test_unseen.loc[img_id]["label"],
            }
    return retrieve_train, retrieve_val, retrieve_test_seen, retrieve_test_unseen


# This function is used for sparse_baseline experiment, sparse_baseline.py
# where training data is used for retrieval.
# For each query in the validation set, we find the top k most similar feature in the training set.
def sparse_retrieval(retrieve_train, retrieve_val, retrieve_size=30):
    """

    Args:
        retrieve_train (dictionary of dictionary): training data for retrieval
        retrieve_val (dictionary of dictionary): validation data for retrieval
        retrieve_size (int, optional): topk for retrieval. Defaults to 30.

    Returns:
        _type_: logging_dict that contains the retrieval results
    """

    # Form the corpus with the training text
    corpus = [retrieve_train[img_id]["text"] for img_id in retrieve_train]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize the empty dic for logging
    logging_dict = EasyDict()

    for id in tqdm(retrieve_val):
        # Get the query text
        query = retrieve_val[id]["text"]
        tokenized_query = query.split(" ")
        # Find the best mathcing documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the largest scores indices
        topk_indices = np.argsort(doc_scores)[::-1][:retrieve_size]
        # Get the image ids from the indices
        topk_ids = [list(retrieve_train.keys())[index]
                    for index in topk_indices]
        # Get the labels
        topk_labels = [retrieve_train[img_id]["label"] for img_id in topk_ids]
        # Get the retrieved scores
        retrieved_scores = [doc_scores[index] for index in topk_indices]
        logging_dict[id] = {
            "no_retrieved": len(topk_ids),
            "retrieved_ids": topk_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": topk_labels,
        }
    return logging_dict

# This function is used for actual RAC: retrieval augmented classification experiment,
# rac_full_sparse4hardnegative.py
# For each query in the training set (batch) during the training process,
# we find the top k most similar feature in the training set, but with opposite labels (hard negatives).


def sparse_retrieve_hard_negatives(retrieve_train, query_ids, retrieve_size=None):
    """
    Args:
        retrieve_train (dictionary of dictionary): training (database) data for retrieval
        query_ids (list of strings): query data for retrieval, using ids as identifiers
        retrieve_size (int, optional): topk for retrieval. Defaults to None,
                                        which means using all the training data. 

    Returns:
        _type_: logging_dict that contains the retrieval results
    """

    # Form the corpus with the training text
    corpus = [retrieve_train[img_id]["text"] for img_id in retrieve_train]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize the empty dic for logging
    logging_dict = EasyDict()

    for id in tqdm(query_ids):
        # Get the query text
        query = retrieve_train[id]["text"]
        label = retrieve_train[id]["label"]
        tokenized_query = query.split(" ")
        # Find the best mathcing documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the scores indices in descending order
        # (The reference of the index is defined by the retrieve_train)
        all_indices = np.argsort(doc_scores)[::-1]
        # Get the image ids from the indices
        # write to a list, thus now we have a list of ids with descending scores
        all_ids = [list(retrieve_train.keys())[index] for index in all_indices]
        # Now that we have the ids, we can get the labels for each items
        all_labels = [retrieve_train[img_id]["label"] for img_id in all_ids]
        # Get the retrieved scores for weighting the loss later on
        retrieved_scores = [doc_scores[index] for index in all_indices]

        # Since we are using hard negatives, we need to remove the positive samples
        # For now use a for loop, but can be optimized later on
        """for i in range(len(all_labels)):
            if all_labels[i] == label:
                all_ids.pop(i)
                all_labels.pop(i)
                retrieved_scores.pop(i)
        """
        all_ids_new = []
        retrieved_scores_new = []
        for i in range(len(all_labels)):
            if all_labels[i] != label:
                all_ids_new.append(all_ids[i])
                retrieved_scores_new.append(retrieved_scores[i])

        # If we have retrieve_size, then we only keep the topk
        if retrieve_size is not None:
            topk_ids = all_ids_new[:retrieve_size]
            # topk_labels = all_labels[:retrieve_size]
            retrieved_scores = retrieved_scores_new[:retrieve_size]
        else:
            topk_ids = all_ids_new
            # topk_labels = all_labels
            retrieved_scores = retrieved_scores_new

        # assert len(topk_ids) == len(retrieved_scores)
        """for i in topk_labels:
            assert i != label"""

        logging_dict[id] = {
            "query_id": id,
            "retrieved_ids": topk_ids,
            "retrieved_scores": retrieved_scores,
            # "retrieved_label": topk_labels,
        }

    return logging_dict


def dense_retrieve_hard_negatives_pseudo_positive(
    train_dl, query_feats, query_labels, model,
    largest_retrieval=1, threshold=None, args=None,
    train_feats=None, train_labels=None,
):
    model.eval()
    # Get the batch size, do not use args.batch_size,
    # since the last batch might be smaller
    batch_size = query_feats.shape[0]
    
    if args.Faiss_GPU == False:
        # For cpu implementation
        query_feats = query_feats.cpu().detach().numpy().astype("float32")
    # print("start to get the training features for retrieval")
    # If we set the train_feats and train_labels to None in upper level,
    # We will reindex the searching index with updated training data
    if train_feats == None or train_labels == None:
        #print("Start to reindex dense retrieval index")
        for i, batch in enumerate(train_dl):
            
            
            if "feats" in batch:
                # Single feature mode
                feats = batch["feats"].to(args.device)
                _, all_feats = model(feats, return_embed=True)
            elif "image_feats" in batch and "text_feats" in batch:
                # Multi-modal feature mode
                image_feats = batch["image_feats"].to(args.device)
                text_feats = batch["text_feats"].to(args.device)
                _, all_feats = model(image_feats, text_feats, return_embed=True)
            else:
                raise ValueError("Unexpected batch structure. Supported keys: 'feats' or ('image_feats', 'text_feats')")
            
            if i == 0:

                if args.Faiss_GPU:
                    # For GPU implementation
                    train_feats = all_feats
                    train_labels = batch["labels"]
                else:
                    # For cpu implementation

                    train_feats = all_feats.cpu().detach().numpy().astype("float32")
                    train_labels = batch["labels"].cpu(
                    ).detach().numpy().astype("int")

            else:

                if args.Faiss_GPU:
                    # For GPU implementation
                    train_feats = torch.cat((train_feats, all_feats), dim=0)
                    train_labels = torch.cat(
                        (train_labels, batch["labels"]), dim=0)
                else:
                    # For cpu implementation
                    train_feats = np.concatenate(
                        (train_feats, all_feats.cpu().detach().numpy().astype("float32")))
                    train_labels = np.concatenate(
                        (train_labels, batch["labels"].cpu().detach().numpy().astype("int")))

    # Perform dense retrieval
    # Get the dimension of the features
    dim = train_feats.shape[1]
    # Initialize the index
    # For different loss functions, we need to change the index type
    if args.metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    # print("start to add the training features to the index")
    if args.Faiss_GPU:
        # print("start to transfer the FAISS index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

        if args.metric != "ip":

            train_feats_normalized = torch.nn.functional.normalize(
                train_feats, p=2, dim=1)
            query_feats_normalized = torch.nn.functional.normalize(
                query_feats, p=2, dim=1)
        else:
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats

    else:
        if args.metric != "ip":
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats
            faiss.normalize_L2(train_feats_normalized)
            faiss.normalize_L2(query_feats_normalized)
        else:
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats

    index.add(train_feats_normalized)

    # Search at most args.hardest_negatives_multiple of the largest retrieval no.

    D, I = index.search(query_feats_normalized,
                        largest_retrieval*args.hard_negatives_multiple)

    # Initialize the hard negative features
    # Each item in the batch has no_hard_negatives hard negatives
    # Thus, this is a 3D tensor of size batch_size x no_hard_negatives x dim
    hard_negative_features = torch.zeros(
        batch_size, args.no_hard_negatives, dim, device="cuda"
    )
    if args.no_pseudo_gold_positives != 0:
        pseudo_positive_features = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, dim, device="cuda")
    # hard_negative_features = query_feats.unsqueeze(1).expand(batch_size,largest_retrieval, -1)

    # Initialize the hard negative retrieved scores
    hard_negative_scores = torch.zeros(
        batch_size, largest_retrieval, device="cuda")
    if args.no_pseudo_gold_positives != 0:
        pseudo_positive_scores = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, device="cuda")

    # Fill the hard_negative_features with the original features multiplied by -1
    # so that in case, we did not find enough hard negatives, we can still use the original features
    # to compute the cos similarity loss.
    # If we do not do this, then the loss will be undefined
    # By doing so, we will get a cosine similarity of -1, which is the perfect dissimilar score,
    # and we thus minimizes the loss (giving some sort of reward in not finding the hard negatives)
    #

    for i, row in enumerate(D):

        # Initialize the counter for the number of hard negatives
        j = 0

        # initalize the counter for the number of pseudo gold positives
        k = 0
        for iter, value in enumerate(row):
            # print(query_labels[i].item())
            # If the label is opposite (negative)
            # print(train_labels[I[i, j]].item(), query_labels[i].item(), query_labels[i], train_labels[I[i, j]].item() != query_labels[i].item())

            # For the hard negatives
            if train_labels[I[i, iter]].item() != query_labels[i].item() and j < args.no_hard_negatives:

                if args.Faiss_GPU:
                    # GPU implementation
                    hard_negative_features[i][j] = train_feats[I[i, iter]]
                    hard_negative_scores[i][j] = value
                else:

                    # CPU implementation with numpy
                    hard_negative_features[i][j] = torch.from_numpy(
                        train_feats[I[i, iter]]).float().to("cuda")
                    hard_negative_scores[i][j] = torch.from_numpy(
                        np.asarray(value)).float().to("cuda")

                j += 1

            # For the pseudo gold positives
            elif train_labels[I[i, iter]].item() == query_labels[i].item() and k < args.no_pseudo_gold_positives:
                if args.Faiss_GPU:
                    # GPU implementation
                    pseudo_positive_features[i][k] = train_feats[I[i, iter]]
                    pseudo_positive_scores[i][k] = value
                else:
                    # CPU implementation with numpy
                    pseudo_positive_features[i][k] = torch.from_numpy(
                        train_feats[I[i, iter]]).float().to("cuda")
                    pseudo_positive_scores[i][k] = torch.from_numpy(
                        np.asarray(value)).float().to("cuda")

                k += 1
            # Only if both the number of hard negatives and pseudo gold positives are found, then break
            if j == largest_retrieval and k == args.no_pseudo_gold_positives:
                break
        # print("Searched top {} to get {} hard negatives".format(iter+1, j))
        
    if args.no_pseudo_gold_positives == 0:
        return hard_negative_features, hard_negative_scores, train_feats, train_labels
    elif args.no_pseudo_gold_positives != 0:
        return hard_negative_features, hard_negative_scores, pseudo_positive_features, pseudo_positive_scores, train_feats, train_labels


def sparse_retrieve_hard_negatives_pseudo_positive(
    ids,
    labels,
    train_set,
    model,
    sparse_retrieval_dictionary,
    args,
):

    all_ids = train_set.ids
    # Give an id as key,
    # the dictionary gives you the index in the trainset
    
    #TODO find new ways to do this 
    #ids2index = train_set.ids_dics
    ids2index = {k: v for v, k in enumerate(all_ids)}
    
    # Train_len * feats
    all_img_features = train_set.image_feats
    all_text_features = train_set.text_feats
    all_labels = train_set.labels
    batch_size = len(ids)
    hard_negative_features = torch.zeros(
        batch_size, args.no_hard_negatives, args.proj_dim, device="cuda"
    )


    pseudo_positive_features = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, args.proj_dim, device="cuda")
    hard_positive_features = torch.zeros(
            batch_size, args.no_hard_positives, args.proj_dim, device="cuda")

    # Interate over all the examples in the batch
    for index_batch, (idx, labelx) in enumerate(zip(ids, labels)):

        # For a sample in the batch,
        # we retrieve the top k most similar feature in the training set
        # K is defined when generating the sparse_retrieval_dictionary

        # Get the retrieved ids and labels for the sample in the batch
        retrieved_id_list = sparse_retrieval_dictionary[idx]["retrieved_ids"]
        retrieved_label_list = sparse_retrieval_dictionary[idx]["retrieved_labels"]

        # Initialize the counter for the number of hard/pseudo
        pseudo_positive_counter = 0
        hard_negative_counter = 0
        if args.sparse_topk == None or args.sparse_topk == -1:
            args.sparse_topk = len(retrieved_id_list)
        for index_topk, (retrieved_id, retrieved_label) in enumerate(zip(retrieved_id_list, retrieved_label_list)):
            # Get the index in the train set to get the features
            index_trainset = ids2index[retrieved_id]
            # Check if the index is correct by checking the label and ids
            assert all_labels[index_trainset] == retrieved_label, \
                "Sparse retrieval label mismatch"
            assert all_ids[index_trainset] == retrieved_id, \
                "Sparse retrieval id mismatch"
            # Encode the features with the model
            # Unsqueeze to add the batch dimension
            model.eval()

            # When same label, psuedo gold
            if retrieved_label == labelx and pseudo_positive_counter < args.no_pseudo_gold_positives:
                _, encoded_feature_x = model(
                    all_img_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    all_text_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    return_embed=True
                )
                pseudo_positive_features[index_batch][pseudo_positive_counter] = encoded_feature_x
                pseudo_positive_counter += 1
            # When different label, hard negative
            elif retrieved_label != labelx and hard_negative_counter < args.no_hard_negatives:
                _, encoded_feature_x = model(
                    all_img_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    all_text_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    return_embed=True
                )
                hard_negative_features[index_batch][hard_negative_counter] = encoded_feature_x
                hard_negative_counter += 1
            # If both hard negatives and pseudo gold positives are found,
            # then break the inside loop
            elif pseudo_positive_counter >= args.no_pseudo_gold_positives and hard_negative_counter >= args.no_hard_negatives:
                break
            elif index_topk >= args.sparse_topk:
                break
        # If not enough hard negatives are found, then just keep zero in the feature space
    return hard_negative_features, pseudo_positive_features
