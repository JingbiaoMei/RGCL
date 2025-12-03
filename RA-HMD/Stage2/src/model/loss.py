import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from utils.retrieval import dense_retrieve_hard_negatives_pseudo_positive, sparse_retrieve_hard_negatives_pseudo_positive


def compute_loss(batch, 
                train_dl, 
                model, 
                args, 
                train_set=None, 
                sparse_retrieval_dictionary=None,
                train_feats=None,
                train_labels=None
                ):
    ids = batch["ids"]
    batch_size = len(ids)

    labels = batch["labels"].to(args.device)
    model.train()
    if "image_feats" in batch and "text_feats" in batch:
        image_feats = batch["image_feats"].to(args.device)
        text_feats = batch["text_feats"].to(args.device)
        output, feats = model(image_feats, text_feats, return_embed=True)

    elif "feats" in batch:
        feats = batch["feats"].to(args.device)
        output, feats = model(feats, return_embed=True)
    else:
        raise ValueError("Unexpected batch structure. Please check your input data.")

    
    # We construct a matrix for label coincidences (Mask matrix for later loss computation)
    # 1 if the labels are the same (positive), 0 otherwise (negative)
    # The dimension would be batch_size x batch_size
    # This is used for the in-batch positive/negative mining

    # We construct it by stacking rows of the labels
    # then for ith row with label 0, we flip the label bit for whole row.

    # We can do this since, if the original label is 0 and
    # the target label is 0, then we have in-batch positive (1);
    # if the target label is 1, then we have in-batch negative (0)
    # Thus we flip the label for 0.

    # We first construct the inverse label, i.e., binary NOT operator on the label
    # Vectors of 1s and 0s of size batch_size
    labels = labels.bool()
    labels_inverse = ~labels
    # print(labels)
    # Matrix of size batch_size x batch_size
    label_matrix = torch.stack(
        [
            labels if labels[i] == True else labels_inverse
            for i in range(batch_size)
        ],
        axis=0,
    )
    # Bool to int conversion
    if args.no_pseudo_gold_positives == 0:
        label_matrix_positive = label_matrix.int()
    # print(label_matrix_positive)
    # FLip
    label_matrix_negative = (~label_matrix).int()
    # print(label_matrix_negative)
    # We then compute the number of in-batch positives and negatives per sample in the batch
    # vectors of sizes batch_size
    # Since the matrix is symmetric, use which dimension does not matter
    # -1 for minus the sample itself

    in_batch_positives_no = torch.sum(label_matrix, dim=1) - 1
    in_batch_negative_no = batch_size - in_batch_positives_no - 1

    # We then construct the similarity matrix by computing the
    # choice of loss function:
    # 1. cosine similarity
    # 2. Triplet loss
    # 3. Manhatten distance

    # We expand the feature matrix to a 3D tensor for vectorized computation
    # feats_expand Dimension: batch_size x feature_size x batch_size
    feats_expanded = feats.unsqueeze(
        2).expand(batch_size, -1, batch_size)

    if args.metric == "cos":
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)

        # We compute the cosine similarity between each pair of features
        sim_matrix = cos(
            feats_expanded, feats_expanded.transpose(0, 2))
    elif args.metric == "ip":
        # might be wrong
        sim_matrix = torch.sum(
            feats_expanded * feats_expanded.transpose(0, 2), dim=1) / args.proj_dim
    elif args.metric == "l2":
        # l2 = nn.PairwiseDistance(p=2, eps=1e-8)
        # Poor vectorized implementation for pairwise distance
        # We use mse instead for vectorized computation

        """
        l2 = torch.nn.MSELoss(reduction='none')
        sim_matrix = l2(
            feats_expanded, feats_expanded.transpose(0, 2)).sum(dim=1) / args.proj_dim
        """
        # sim_matrix = torch.sum(torch.square((feats_expanded - feats_expanded.transpose(0, 2))), dim=1)
        sim_matrix = compute_l2(feats_expanded, feats_expanded.transpose(
            0, 2), normalise=args.norm_feats_loss, sum_dim=1, sqrt=args.l2_sqrt)
        # Add a negative sign here to account for the fact that
        # L2 is a distance measure, not a similarity measure, larger is more distant (dissimilar),
        # Where in similarity measure, larger is more similar

        # SQRT here gives NAN, thus we minimize the square of the L2 distance
        sim_matrix = - sim_matrix / args.proj_dim

    # The diagonal of the similarity matrix is 1,
    # which is the similarity of the same pair
    # Thus replace it with 0
    sim_matrix.fill_diagonal_(0)

    # We compute the loss matrix by multiplying the similarity matrix
    in_batch_negative_loss = sim_matrix * label_matrix_negative

    if args.no_pseudo_gold_positives == 0:
        in_batch_positives_loss = sim_matrix * label_matrix_positive
    else:
        # If we use pseudo gold positives, we do not use in batch positives
        # We set it to a matrix of zeros to make sure the contrastive loss can still use the same code
        in_batch_positives_loss = torch.zeros(
            batch_size, batch_size).to(args.device)
    # Wrong implementation since division by zero might happen
    """in_batch_loss = torch.sum(in_batch_negative_loss) / torch.sum(
        in_batch_negative_no
    ) - torch.sum(in_batch_positives_loss) / torch.sum(in_batch_positives_no)"""

    # We compute the loss by summing over the loss matrix

    # V1 implementation
    #  If there is no in_batch_negative, we set the loss to 0 to avoid division by zero
    """    if torch.sum(in_batch_negative_no) == 0:
        in_batch_negative_loss_sum = 0
    else:
        in_batch_negative_loss_sum = torch.sum(
            in_batch_negative_loss
        ) / torch.sum(in_batch_negative_no) 
    """

    # V2 implementation
    """in_batch_negative_loss_sum = torch.zeros(batch_size).to(args.device)
    in_batch_positives_loss_sum = torch.zeros(batch_size).to(args.device)
    for i in range(batch_size):
        if in_batch_negative_no[i] == 0:
            in_batch_negative_loss_sum[i] = 0
        else:
            in_batch_negative_loss_sum[i] = torch.sum(in_batch_negative_loss[i]) / in_batch_negative_no[i]"""

    # V3 implementation masking and replacing nan by zero
    """
    ## in_batch_negative_loss_sum = torch.sum(in_batch_negative_loss, dim=1) / in_batch_negative_no
    
    # Assert the number of samples from in_batch_negative_no and the number of non zero elements in in_batch_negative_loss are the same
    mask = in_batch_negative_loss != 0
    in_batch_negative_loss_sum = torch.sum(in_batch_negative_loss, dim=1) / mask.sum(dim=1)
    # Check if there is nan, if so replace it by zero
    in_batch_negative_loss_sum[torch.isnan(in_batch_negative_loss_sum)] = 0
    """
    # V4 implementation, doing the loss with the mask rather than detect if there is nan and set to zero:
    # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
    neg_mask = in_batch_negative_loss != 0

    # Dim batch_size, count the number of zeros for each sample in the batch,
    neg_zero_count = (neg_mask == 0).sum(dim=1)

    # However, if all the terms are zero, we will get nan due to zero division,
    # We will form a further mask to only operate on the sample with at least one non-zero term
    neg_zero_count_zero_mask = torch.zeros(batch_size, device=args.device) != in_batch_negative_no

    in_batch_negative_loss_sum = torch.zeros(batch_size, device=args.device)
    in_batch_negative_loss_sum[neg_zero_count_zero_mask] = torch.sum(
        in_batch_negative_loss[neg_zero_count_zero_mask], dim=1) / neg_mask.sum(dim=1)[neg_zero_count_zero_mask]

    # Only use in-batch positive if we do not use pseudo gold positive samples
    if args.no_pseudo_gold_positives == 0:
        # V1 implementation
        """if torch.sum(in_batch_positives_no) == 0:
            in_batch_positives_loss_sum = 0
        else:
            in_batch_positives_loss_sum = torch.sum(
                in_batch_positives_loss
            ) / torch.sum(in_batch_positives_no) """

        # V2 implementation
        """for i in range(batch_size):
            if in_batch_positives_no[i] == 0:
                in_batch_positives_loss_sum[i] = 0
            else:
                in_batch_positives_loss_sum[i] = torch.sum(in_batch_positives_loss[i]) / in_batch_positives_no[i]"""

        """# V3 implementation masking and replacing nan by zero
        ## in_batch_positives_loss_sum = torch.sum(in_batch_positives_loss, dim=1) / in_batch_positives_no
        mask = in_batch_positives_loss != 0
        in_batch_positives_loss_sum = torch.sum(in_batch_positives_loss, dim=1) / mask.sum(dim=1)
        # Check if there is nan, if so replace it by zero
        in_batch_positives_loss_sum[torch.isnan(in_batch_positives_loss_sum)] = 0"""
        # V4 implementation, doing the loss with the mask rather than detect if there is nan and set to zero:
        # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
        pos_mask = in_batch_positives_loss != 0
        # Dim batch_size, count the number of zeros for each sample in the batch,
        pos_zero_count = (pos_mask == 0).sum(dim=1)
        # However, if all the terms are zero, we will get nan due to zero division,
        # We will form a further mask to only operate on the sample with at least one non-zero term
        pos_zero_count_zero_mask = pos_zero_count != in_batch_positives_no

        in_batch_positives_loss_sum = torch.zeros(
            batch_size, device=args.device)
        in_batch_positives_loss_sum[pos_zero_count_zero_mask] = torch.sum(
            in_batch_positives_loss[pos_zero_count_zero_mask], dim=1) / pos_mask.sum(dim=1)[pos_zero_count_zero_mask]

    # If we use pseudo gold positives, we do not use in batch positives
    else:
        in_batch_positives_loss_sum = 0
    if args.in_batch_loss:
        in_batch_loss = in_batch_negative_loss_sum - in_batch_positives_loss_sum
    else: 
        in_batch_loss = torch.Tensor([0.0]).to(args.device).to(feats.dtype)
    # Sanity check

    """print("feature vector")
    print(feats.shape),
    print(feats)
    print("in-batch loss")
    print(in_batch_negative_loss.shape,
        in_batch_negative_no.shape, in_batch_negative_loss, in_batch_negative_no, torch.mean(in_batch_negative_loss, dim=1))
    
    print(in_batch_positives_loss.shape,
        in_batch_positives_no.shape, in_batch_positives_loss, in_batch_positives_no, torch.mean(in_batch_positives_loss, dim=1))
    
    print("in-batch positive loss sum:", in_batch_positives_loss_sum)
    print("in-batch negative loss sum:", in_batch_negative_loss_sum)
    print("in-batch loss sum:", in_batch_loss)
    """

    # ----------------- Hard Negative Retrieval and Pseudo Gold Positive -----------------
    # retrieve hard negatives and pseudo gold with Dense retrieval

    # Only hard negative
    #print("start to retrieve hard negatives and pseudo gold")
    if args.sparse_dictionary is None:
        if args.hard_negatives_loss and args.no_pseudo_gold_positives == 0:
            (
                hard_negative_features,
                hard_negative_scores,
                train_feats, 
                train_labels,
            ) = dense_retrieve_hard_negatives_pseudo_positive(
                train_dl,
                feats,
                labels,
                model,
                largest_retrieval=args.no_hard_negatives,
                args=args,
                train_feats=train_feats,
                train_labels=train_labels,
            )
        # Both hard negative and pseudo gold,
        # In default we will consider hard negative, which is key
        # to the good performance. 
        # But if we want to test without hard negative, this is also fine
        # We can just ignore the hard negative features and scores
        elif args.no_pseudo_gold_positives > 0:
            (
                hard_negative_features,
                hard_negative_scores,
                pseudo_positive_features,
                pseudo_positive_scores,
                train_feats, 
                train_labels,
            ) = dense_retrieve_hard_negatives_pseudo_positive(
                train_dl,
                feats,
                labels,
                model,
                largest_retrieval=args.no_pseudo_gold_positives,
                args=args,
                train_feats=train_feats,
                train_labels=train_labels,
            )
        else:
            pass
    # For sparse retrieval, 
    # we always retrieve both hard negatives and pseudo gold
    # Since no computation will be saved 
    # by only retrieving hard negatives/pseudo gold
    else:
        (   hard_negative_features,
            pseudo_positive_features,   
        )= sparse_retrieve_hard_negatives_pseudo_positive(
            ids,
            labels,
            train_set,
            model,
            sparse_retrieval_dictionary,
            args,
        )
                
            

    # for hard negative loss
    if args.hard_negatives_loss:
        # Now we have the hard negatives features, we compute the loss

        # hard_negative_scores size batch_size, largest_retrieval

        # We compute the similarity matrix between the hard negatives and the original features
        # The dimension of hard_negative_features is batch_size x no_hard_negatives x dim
        # The dimension of original feats is batch_size x dim
        # We thus need to expand the original feats to batch_size x no_hard_negatives x embed_dim/hidden_dim
        feats_expanded = feats.unsqueeze(1).expand(
            batch_size, args.no_hard_negatives, -1
        )

        # The returned hard_negative_features might contain all zero embeddings for some samples,
        # We need to discard them in the loss computation
        # What we need to do is to construct a mask to zero out the loss for those samples

        # For simplicity, we only check if the first dimension is zero in the feature embedding
        # The mask is batch_size x no_hard_negatives, 1 if embedding non zero, 0 if embedding zero,
        # Thus we can multiply the mask with the loss.
        #zeroLoss_mask = hard_negative_features[:, :, 0] != 0

        # 2024.12.07 update, the above method is not correct, since the first dimension can be zero for some samples
        # Instead, we will sum the sum of the value of the embeddings
        # If the sum is zero, then we will set the mask to zero
        zeroLoss_mask = torch.sum(hard_negative_features, dim=2) != 0

        # Compute Hard negative loss, at the third dimension feature dimension(dim=2)

        if args.metric == "cos":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            # print(hard_negative_scores)
            # we compute the cosine similarity
            cos_hard = nn.CosineSimilarity(dim=2, eps=1e-8)
            hard_loss = zeroLoss_mask * cos_hard(
                feats_expanded, hard_negative_features)
            # print(hard_loss.shape)
            # print(hard_loss)
        elif args.metric == "ip":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            hard_loss = zeroLoss_mask * torch.sum(
                feats_expanded * hard_negative_features, dim=2
            ) / args.proj_dim

        elif args.metric == "l2":

            """
            l2_hard = torch.nn.MSELoss(reduction='none')
            hard_loss = l2_hard(feats_expanded,
                                hard_negative_features).sum(dim=2)
            """
            # hard_loss = zeroLoss_mask * torch.sum(torch.square((feats_expanded - hard_negative_features)), dim=2)
            hard_loss = compute_l2(feats_expanded, hard_negative_features,
                                   normalise=args.norm_feats_loss, sum_dim=2, sqrt=args.l2_sqrt)
            hard_loss *= zeroLoss_mask
            """print("feats_expanded:", feats_expanded)
            print("hard negative features:", hard_negative_features)
            print("hard negative features shape:", hard_negative_features.shape)
            print("hard loss:", hard_loss)"""
            # SQRT gives NAN, thus we minimize the square of the L2 distance
            hard_loss = - hard_loss / args.proj_dim

        # For contrastive loss, we take mean during the loss computation
        if args.loss != "contrastive":
            # Hard loss batch_size * no_hard_neg -> batch_size
            hard_loss = torch.sum(hard_loss, dim=1)
            """print("hard loss")
            print(hard_loss.shape)
            print(hard_loss)"""

    # If not using hard negative, set to 0
    else:
        #hard_loss = 0
        hard_loss = torch.tensor([0.0], device=args.device)

    # for pseudo gold loss
    if args.no_pseudo_gold_positives != 0:
        # Now we have the pseudo gold positive features, we compute the loss
        # pseudo_positive_scores size: batch_size, args.no_pseudo_gold_positives

        feats_expanded = feats.unsqueeze(1).expand(
            batch_size, args.no_pseudo_gold_positives, -1
        )
        if args.metric == "cos":
            # Compute loss
            # Loss is batch_size x no_pseudo_gold_positives
            # print(pseudo_positive_scores)
            # we compute the cosine similarity
            cos_pseudo_gold = nn.CosineSimilarity(dim=2, eps=1e-8)
            pseudo_gold_loss = cos_pseudo_gold(
                feats_expanded, pseudo_positive_features)
            # print(pseudo_gold_loss.shape)
            # print(pseudo_gold_loss)
        elif args.metric == "ip":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            pseudo_gold_loss = torch.sum(
                feats_expanded * pseudo_positive_features, dim=2
            ) / args.proj_dim

        elif args.metric == "l2":

            # pseudo_gold_loss = torch.sum(torch.square((feats_expanded - pseudo_positive_features)), dim=2)
            pseudo_gold_loss = compute_l2(
                feats_expanded, pseudo_positive_features, normalise=args.norm_feats_loss, sum_dim=2, sqrt=args.l2_sqrt)
            """print("feats_expanded:", feats_expanded)
            print("Pseudo Positive Feats:", pseudo_positive_features)
            print("Pseudo Positive Feats Shape:", pseudo_positive_features.shape)
            print("Pseiudo Gold Loss:", pseudo_gold_loss)"""
            # SQRT gives NAN, thus we minimize the square of the L2 distance
            pseudo_gold_loss = - pseudo_gold_loss / args.proj_dim

        # For contrastive loss, we take mean during the loss computation
        if args.loss != "contrastive":

            pseudo_gold_loss = torch.mean(pseudo_gold_loss, dim=1)
            """print("pseudo_gold loss")
            print(pseudo_gold_loss.shape)
            print(pseudo_gold_loss)"""

    # if not using psedo gold, set to 0
    else:
        #pseudo_gold_loss = 0
        # use tensor zero instead
        pseudo_gold_loss = torch.tensor([0.0], device=args.device)

    if args.loss == "naive":
        # Take mean on batch-sample level
        total_loss = torch.mean(in_batch_loss + hard_loss - pseudo_gold_loss)
    elif args.loss == "triplet":
        total_loss = torch.mean(torch.relu(
            in_batch_loss + hard_loss - pseudo_gold_loss + args.triplet_margin))

        # Don't use if statement, rather, we can use a relu
        # if total_loss < 0:
        #    total_loss = torch.tensor([0.0], requires_grad=True).to(args.device)
    elif args.loss == "contrastive":

        # Dim Batch size * Batch size
        # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
        neg_mask = in_batch_negative_loss != 0
        # Dim batch_size, count the number of zeros for each sample in the batch,
        # Since exponential of zero gives 1, we will delete the the number of zeros to discard the zero term
        neg_zero_count = (neg_mask == 0).sum(dim=1)
        # However, if all the terms are zero, we will get nan due to zero division,
        # We will form a further mask to only operate on the sample with at least one non-zero term
        # neg_zero_count_zero_mask = neg_zero_count == 0
        # Above is incorrect, we need to get the mask for samples in the batch with all examples zero
        neg_zero_count_zero_mask = torch.zeros(batch_size, device=args.device) != in_batch_negative_no
        in_batch_negative_loss_tmp = torch.zeros(
            batch_size, device=args.device)
        #in_batch_negative_loss_tmp[neg_zero_count_zero_mask] = (torch.exp(in_batch_negative_loss[neg_zero_count_zero_mask]).sum(
        #    dim=1) - neg_zero_count[neg_zero_count_zero_mask]) / (neg_mask.sum(dim=1))[neg_zero_count_zero_mask]
        
        in_batch_negative_loss_tmp[neg_zero_count_zero_mask] = (torch.exp(in_batch_negative_loss[neg_zero_count_zero_mask]).sum(
            dim=1) - neg_zero_count[neg_zero_count_zero_mask])
        in_batch_negative_loss = in_batch_negative_loss_tmp
        """print(in_batch_negative_no)
        print(neg_zero_count)
        print(neg_zero_count_zero_mask)
        print(neg_mask.sum(dim=1))
        print((neg_mask.sum(dim=1))[neg_zero_count_zero_mask])"""

        if args.no_hard_negatives != 0:
            # Dim batch size x no_hard_negatives
            hard_neg_mask = hard_loss != 0

            # Dim batch_size
            hard_zero_count = (hard_neg_mask == 0).sum(dim=1)
            # Constract this matrix to avoid zero division error
            # hard_zero_count_zero_mask = hard_zero_count == 0
            hard_zero_count_zero_mask = hard_zero_count != args.no_hard_negatives
            # initialise all zero matrix for hard loss
            hard_loss_tmp = torch.zeros(batch_size, device=args.device)
            # We need to count the number of zero terms to discard them in the loss computation,
            # Since zero terms gives exp(0) = 1, we will delete the the number of zeros to discard the zero term
            hard_loss_tmp[hard_zero_count_zero_mask] = (torch.exp(hard_loss[hard_zero_count_zero_mask]).sum(
                dim=1) - hard_zero_count[hard_zero_count_zero_mask]) / (hard_neg_mask.sum(dim=1))[hard_zero_count_zero_mask]
            hard_loss = hard_loss_tmp

        """print(hard_zero_count)
        print(hard_zero_count_zero_mask)
        print((hard_neg_mask.sum(dim=1)))
        print((hard_neg_mask.sum(dim=1))[hard_zero_count_zero_mask])"""

        # If we dont have pseudo gold positives, we use the in batch positives
        if args.no_pseudo_gold_positives == 0:

            """loss = - torch.log(torch.mean(torch.exp(in_batch_positives_loss), dim=1) / (torch.mean(torch.exp(in_batch_negative_loss),
                            dim=1) + torch.mean(torch.exp(in_batch_positives_loss), dim=1) + torch.mean(torch.exp(hard_loss), dim=1)))"""
            """pos_mask = in_batch_positives_loss != 0
            pos_zero_count = (pos_mask == 0).sum(dim=1)

            pos_zero_count_zero_mask = pos_zero_count != in_batch_positives_no

            in_batch_positives_loss_tmp = torch.zeros(
                batch_size, device=args.device)
            in_batch_positives_loss_tmp[pos_zero_count_zero_mask] = (torch.exp(in_batch_positives_loss[pos_zero_count_zero_mask]).sum(
                dim=1) - pos_zero_count[pos_zero_count_zero_mask]) / (pos_mask.sum(dim=1))[pos_zero_count_zero_mask]
            in_batch_positives_loss = in_batch_positives_loss_tmp"""
            in_batch_positives_loss = torch.mean(torch.exp(in_batch_positives_loss), dim=1)
            loss = - torch.log(in_batch_positives_loss /
                               (in_batch_negative_loss + in_batch_positives_loss + hard_loss))
        # If we have pseudo gold positives, we use the pseudo gold positives rather than the in batch positives
        else:

            """loss = - torch.log(torch.mean(torch.exp(pseudo_gold_loss), dim=1) / (torch.mean(torch.exp(hard_loss),
                            dim=1) + torch.mean(torch.exp(pseudo_gold_loss), dim=1) + torch.mean(torch.exp(in_batch_negative_loss), dim=1)))"""

            pseudo_gold_loss = torch.mean(torch.exp(pseudo_gold_loss), dim=1)

            loss = - torch.log(pseudo_gold_loss / (hard_loss +
                               pseudo_gold_loss + in_batch_negative_loss))

        """print("Loss:", loss) 
        print("Hard Loss:", hard_loss)
        #print("In Batch Positives Loss:", in_batch_positives_loss)
        print("In Batch Negative Loss:", in_batch_negative_loss)
        print("Pseudo Gold Loss:", pseudo_gold_loss)"""

        total_loss = torch.mean(loss)
    if args.hybrid_loss:
        if args.pos_weight_value != None:
            lossFn_classifier = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([args.pos_weight_value], device=args.device))
        else:
            lossFn_classifier = nn.BCEWithLogitsLoss()
        loss_classifier = lossFn_classifier(
                output, labels.float().reshape(-1, 1))
        #total_loss = (total_loss + loss_classifier * args.ce_weight) / (1 + args.ce_weight)
        total_loss = total_loss * (1-args.ce_weight) + loss_classifier * args.ce_weight
    else:
        loss_classifier = 0

    return total_loss, torch.mean(in_batch_loss), torch.mean(hard_loss), torch.mean(pseudo_gold_loss), loss_classifier, train_feats, train_labels


def compute_l2(feats_1, feats_2, normalise=False, sum_dim=1, sqrt=False, eps=1e-5):
    """Compute L2 loss."""
    l2_loss = 0
    if normalise:
        feats_1 = torch.nn.functional.normalize(feats_1, dim=sum_dim)
        feats_2 = torch.nn.functional.normalize(feats_2, dim=sum_dim)
    if not sqrt:
        l2_loss = torch.sum(torch.square((feats_1 - feats_2)), dim=sum_dim)
    else:
        l2_loss = torch.sqrt(torch.sum(torch.square(
            (feats_1 - feats_2)), dim=sum_dim) + torch.finfo(torch.float32).tiny)

    return l2_loss


def compute_ip(feats_1, feats_2, normalise=False, sum_dim=1):
    return None
