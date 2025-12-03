import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import pickle
import wandb
import json
import torchmetrics
def metrics(dataset="FB", gt_path="./data/gt/", path="./data/experiment/"):
    """
    metrics 
    """
    # Read Ground truth
    if dataset == "MM":
        dev_df = pd.read_json(os.path.join(
            gt_path, 'MM/dev_seen.jsonl'), lines=True)
        test_unseen_df = pd.read_json(os.path.join(
            gt_path, 'MM/test_unseen.jsonl'), lines=True)
        test_df = pd.read_json(os.path.join(
            gt_path, 'MM/test_seen.jsonl'), lines=True)

    # If using downsampled MM, the ground truth for validation is the same
    if dataset == "MM_Down":
        dev_df = pd.read_json(os.path.join(
            gt_path, 'MM/dev_seen.jsonl'), lines=True)
        test_unseen_df = pd.read_json(os.path.join(
            gt_path, 'MM/test_unseen.jsonl'), lines=True)
        test_df = pd.read_json(os.path.join(
            gt_path, 'MM/test_seen.jsonl'), lines=True)

    if dataset == "FB":
        dev_df = pd.read_json(os.path.join(
            gt_path, 'FB/dev_seen.jsonl'), lines=True)
        test_unseen_df = pd.read_json(os.path.join(
            gt_path, 'FB/test_unseen.jsonl'), lines=True)
        test_df = pd.read_json(os.path.join(
            gt_path, 'FB/test_seen.jsonl'), lines=True)

    # Make sure the lists will be ordered, i.e. test[0] is the same model as devs[0]
    dev, test, test_unseen = [], [], []
    # Never dynamically add to a pd Dataframe
    dev_probas, test_probas, test_unseen_probas = {}, {}, {}
    dev_label, test_label, test_unseen_label = {}, {}, {}

    # iterate over all the csv files in the path
    for csv in sorted(os.listdir(path)):

        if ".csv" in csv:

            # If the csv is a dev_seen file
            if ("dev_seen" in csv):
                dev.append(pd.read_csv(os.path.join(path, csv)))
                dev_probas[csv] = pd.read_csv(
                    os.path.join(path, csv)).proba.values
                dev_probas = pd.DataFrame(dev_probas)
                
                dev_label[csv] = pd.read_csv(
                    os.path.join(path, csv)).label.values
                dev_label = pd.DataFrame(dev_label)

                roc = roc_auc_score(dev_df.label, dev_probas[csv])
                acc = accuracy_score(dev_df.label, dev_label[csv])
                pre = precision_score(dev_df.label, dev_label[csv])
                recall = recall_score(dev_df.label, dev_label[csv])
                f1 = f1_score(dev_df.label, dev_label[csv])
                print(csv, "AUROC: {0} , Acc: {1}, Precision: {2}, Recall: {3}, F1: {4}".format(
                    round(roc, 4), round(acc, 4), round(pre, 4), round(recall, 4), round(f1, 4)))
                logging = {"AUROC": roc, "ACC": acc, "Precision": pre,  "Recall": recall, "F1": f1}
                logging = pd.DataFrame(data=logging, index=[csv])
                logging.to_csv(path+"metrics"+csv, mode='w', header=True, index=True)
            
            # If the csv is a test_seen file
            elif "test_unseen" in csv:

                test_unseen.append(pd.read_csv(os.path.join(path, csv)))
                test_unseen_probas[csv] = pd.read_csv(
                    os.path.join(path, csv)).proba.values
                test_unseen_probas = pd.DataFrame(test_unseen_probas)

                test_unseen_label[csv] = pd.read_csv(
                    os.path.join(path, csv)).label.values
                test_unseen_label = pd.DataFrame(test_unseen_label)

                roc = roc_auc_score(test_unseen_df.label,
                                    test_unseen_probas[csv])
                acc = accuracy_score(test_unseen_df.label,
                                     test_unseen_label[csv])
                pre = precision_score(
                    test_unseen_df.label, test_unseen_label[csv])
                recall = recall_score(
                    test_unseen_df.label, test_unseen_label[csv])
                f1 = f1_score(test_unseen_df.label, test_unseen_label[csv])
                print(csv, "AUROC: {0} , Acc: {1}, Precision: {2}, Recall: {3}, F1: {4}".format(
                    round(roc, 4), round(acc, 4), round(pre, 4), round(recall, 4), round(f1, 4)))
                logging = {"AUROC": roc, "ACC": acc, "Precision": pre,  "Recall": recall, "F1": f1}
                logging = pd.DataFrame(data=logging, index=[csv])
                logging.to_csv(path+"metrics"+csv, mode='w', header=True, index=True)
            
            # If the csv is a test_seen file
            elif "test_seen" in csv:

                test.append(pd.read_csv(os.path.join(path, csv)))
                # Get the probabilities
                test_probas[csv] = pd.read_csv(
                    os.path.join(path, csv)).proba.values
                test_probas = pd.DataFrame(test_probas)
                # Get the labels
                test_label[csv] = pd.read_csv(
                    os.path.join(path, csv)).label.values
                test_label = pd.DataFrame(test_label)

                roc = roc_auc_score(test_df.label, test_probas[csv])
                acc = accuracy_score(test_df.label, test_label[csv])
                pre = precision_score(test_df.label, test_label[csv])
                recall = recall_score(test_df.label, test_label[csv])
                f1 = f1_score(test_df.label, test_label[csv])
                print(csv, "AUROC: {0} , Acc: {1}, Precision: {2}, Recall: {3}, F1: {4}".format(
                    round(roc, 4), round(acc, 4), round(pre, 4), round(recall, 4), round(f1, 4)))
                logging = {"AUROC": roc, "ACC": acc, "Precision": pre,  "Recall": recall, "F1": f1}
                logging = pd.DataFrame(data=logging, index=[csv])
                logging.to_csv(path+"metrics"+csv, mode='w', header=True, index=True)

# Define the evaluation function
def evaluate(test_dl, model, name="dev"):
    """
    evaluate the model on the val/test set
    input: val/test dataloader, model, name of the set
    output: print the evaluation metrics
    
    """
    lossFn_classifier = nn.BCEWithLogitsLoss()
    model.eval()
    #print("Start to do evaluation")
    with torch.no_grad():
        actual_probs = np.array([])
        predictions = np.array([])
        labels = np.array([])
        losses = np.array([])
        for step, batch in enumerate((test_dl)):
            images, texts, label,_ = batch 
        
            #texts = clip.tokenize(texts,truncate=True)
            predicted,_,_ = model(images, texts)
            loss = lossFn_classifier(
                predicted, label.reshape(-1, 1).to("cuda").type(torch.float32))
            losses = np.concatenate((losses, loss.reshape(-1).detach().cpu().numpy()))
            actual_probs = np.concatenate((actual_probs, torch.sigmoid(predicted).reshape(-1).detach().cpu().numpy()))
            predictions = np.concatenate((predictions, predicted.reshape(-1).detach().cpu().numpy()))
            labels = np.concatenate((labels, label.detach().cpu().numpy()))
            
        acc = accuracy_score(labels, actual_probs.round())
        roc = roc_auc_score(labels, predictions)
        pre = precision_score(labels, actual_probs.round())
        recall = recall_score(labels, actual_probs.round())
        f1 = f1_score(labels, actual_probs.round())
        loss = np.mean(losses)
        print("Acc: {}\t AUROC: {}\t Precison: {}\t Recall: {}\t F1: {}".format(acc, roc, pre, recall, f1))
        if name == "dev_seen":
            test_metrics = {name+"/loss":loss,
                            name+"/Acc":acc,
                            name+"/AUROC":roc,
                            name+"/Precision":pre,
                            name+"/Recall":recall,
                            name+"/F1":f1}
        else:
            test_metrics = {name+"/Acc":acc,
                            name+"/AUROC":roc,
                            name+"/Precision":pre,
                            name+"/Recall":recall,
                            name+"/F1":f1}
    return actual_probs.round(), actual_probs, predictions, test_metrics    

# Count the trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics_retrieval_baseline(logging_dict, labels, majority_voting="mean"):
    """
    # Loop through the logging_dict to get the retrieved labels
    # Structure of logging_dict:
    # logging_dict[q_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }
    # Try several majority voting methods
    # """
    list_majority_voted = []
    for key, value in logging_dict.items():
        list_majority_voted.append(np.mean(value["retrieved_label"]))
    #print(list_majority_voted)
    try:
        labels = labels.detach().cpu().numpy()
    except:
        labels = labels
    roc = roc_auc_score(labels, list_majority_voted)
    list_majority_voted = (np.array(list_majority_voted)>=0.5)*1
    acc = np.mean(list_majority_voted == labels)
    pre = precision_score(labels, list_majority_voted)
    recall = recall_score(labels, list_majority_voted)
    f1 = f1_score(labels, list_majority_voted)    
    
    #print("accuracy:",acc)
    #print("AUROC:", roc)
    return acc, roc, pre, recall, f1


def compute_metrics_retrieval(logging_dict, labels, majority_voting="mean", topk=0, use_prob=False, use_sim=False):
    """
    # Loop through the logging_dict to get the retrieved labels
    # Structure of logging_dict:
    # logging_dict[q_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }
    # Try several majority voting methods
    # """
    list_majority_voted = []
    list_majority_voted_prob = []
    if majority_voting == "arithmetic":
        weight = np.arange(1, topk+1)
        # Reverse the weight, so the first retrieved one has the highest weight
        weight = weight[::-1]
    if not use_prob and not use_sim:
        for key, value in logging_dict.items():
            retrieved_labels = value["retrieved_label"]
            length = len(retrieved_labels)
            if majority_voting == "mean":
                list_majority_voted.append(np.mean(retrieved_labels))
            elif majority_voting == "arithmetic":
                # taking the :length since we have both topk and threshold to decide the length
                list_majority_voted.append(np.sum(np.array(retrieved_labels)*weight[:length])/np.sum(weight[:length]))
            else:
                raise ValueError("The majority voting method is not supported")
    elif use_prob:
        for key, value in logging_dict.items():
            retrieved_out = value["retrieved_out"]
            length = len(retrieved_out)
            if majority_voting == "mean":
                list_majority_voted_prob.append(torch.mean(retrieved_out).item())
            elif majority_voting == "arithmetic":
                # taking the :length since we have both topk and threshold to decide the length
                weight_t = torch.arange(1, topk+1)
                # Reverse the weight, so the first retrieved one has the highest weight
                weight_t = torch.flip(weight_t, dims=[0])
                list_majority_voted_prob.append((torch.sum(retrieved_out*weight_t[:length])/torch.sum(weight_t[:length])).item())
            retrieved_labels = value["retrieved_label"]
            length = len(retrieved_labels)
            if majority_voting == "mean":
                list_majority_voted.append(np.mean(retrieved_labels))
            elif majority_voting == "arithmetic":
                # taking the :length since we have both topk and threshold to decide the length
                list_majority_voted.append(np.sum(np.array(retrieved_labels)*weight[:length])/np.sum(weight[:length]))
    elif use_sim:
        for key, value in logging_dict.items():
            retrieved_labels = value["retrieved_label"]
            retrieved_sims = value["retrieved_scores"]
            retrieved_sims = np.array([sim.item() for sim in retrieved_sims ])
            #map 0 to -1, map 1 to 1 
            retrieved_labels_map = np.array(retrieved_labels)*2-1
            # times the similarity
            retrieved_labels_map = retrieved_labels_map*retrieved_sims
            length = len(retrieved_labels_map)
            if majority_voting == "mean":
                list_majority_voted.append(np.mean(retrieved_labels_map))
            elif majority_voting == "arithmetic":
                # taking the :length since we have both topk and threshold to decide the length
                list_majority_voted.append(np.sum(np.array(retrieved_labels_map)*weight[:length])/np.sum(weight[:length]))
            else:
                raise ValueError("The majority voting method is not supported") 
    #print(list_majority_voted)
    try:
        labels = labels.detach().cpu().numpy()
    except:
        labels = labels
        
    if not use_prob:
        roc = roc_auc_score(labels, list_majority_voted)
    else:
        roc = roc_auc_score(labels, list_majority_voted_prob)
    if not use_sim:
        list_majority_voted_round = (np.array(list_majority_voted)>=0.5)*1
    else:
        list_majority_voted_round = (sigmoid(np.array(list_majority_voted))>=0.5)*1
    acc = np.mean(list_majority_voted_round == labels)
    pre = precision_score(labels, list_majority_voted_round)
    recall = recall_score(labels, list_majority_voted_round)
    f1 = f1_score(labels, list_majority_voted_round)    
    
    #print("accuracy:",acc)
    #print("AUROC:", roc)
    return acc, roc, pre, recall, f1, list_majority_voted, labels





def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_metrics_retrieval_augmented(logging_dict, labels, majority_voting="mean", vote_before_sigmoid=True, topk=0):
    '''
    "no_retrieved": no_retrieved,
    "retrieved_ids": retrieved_ids,
    "retrieved_scores": retrieved_scores,
    "retrieved_label": retrieved_label,
    "retrieved_out": retrieved_out,
    "eval_out": eval_out[i].cpu().detach(),
    
    '''
    
    
    list_majority_voted = []
    if majority_voting == "arithmetic":
        weight = np.arange(1, topk+1)
        # Reverse the weight, so the first retrieved one has the highest weight
        weight = weight[::-1]
        
        
    for key, value in logging_dict.items():
        retrieved_labels = value["retrieved_label"]
        retrieved_out = value["retrieved_out"]
        length = len(retrieved_labels)
        if majority_voting == "mean":
            if vote_before_sigmoid:
                pred = torch.mean(retrieved_out)
                pred = torch.sigmoid(pred)
                list_majority_voted.append(pred.item())
            else:
                pred = torch.mean(torch.sigmoid(retrieved_out))
                list_majority_voted.append(pred.item())
            
        elif majority_voting == "arithmetic":
            retrieved_out = retrieved_out.cpu().detach().numpy()
            # taking the :length since we have both topk and threshold to decide the length
            if vote_before_sigmoid:
                list_majority_voted.append( sigmoid(np.sum(retrieved_out*weight[:length])/np.sum(weight[:length])))
            else:
                retrieved_out = sigmoid(retrieved_out)
                list_majority_voted.append(np.sum(retrieved_out*weight[:length])/np.sum(weight[:length]))
            
        else:
            raise ValueError("The majority voting method is not supported")
    try:
        labels = labels.detach().cpu().numpy()
    except:
        labels = labels
    roc = roc_auc_score(labels, list_majority_voted)
    list_majority_voted_round = (np.array(list_majority_voted)>=0.5)*1
    acc = np.mean(list_majority_voted_round == labels)
    pre = precision_score(labels, list_majority_voted_round)
    recall = recall_score(labels, list_majority_voted_round)
    f1 = f1_score(labels, list_majority_voted_round)  
    return acc, roc, pre, recall, f1, list_majority_voted, labels


    
# Evaluation for linear probing after retrieval training 
def evaluate_linear_probe(args, test_dl, model, compute_loss=False, name="dev_seen", epoch=0):
    model.eval()
    if compute_loss:
        lossFn_classifier = nn.BCEWithLogitsLoss()
    _, labels, predicted,_ = iterate_dl(args,test_dl, model)
  
    acc, roc, pre, recall, f1, loss = eval_metrics(labels, predicted, name=name, epoch=epoch, compute_loss=compute_loss)

    return acc, roc, pre, recall, f1


# Code for evaluation for liner probe

def iterate_dl(args, dl, classifier):
    # A function to iterate through the dataloader and get all the
    # ids, labels, and predicted labels
    with torch.no_grad():
        ids = []
        labels, predicted, embed = [], [], []

        for step, batch in enumerate(dl):
            ids.extend(batch["ids"])
            batch_labels = batch["labels"].detach().cpu()
            labels.append(batch_labels)

            if "feats" in batch:
                # Single feature mode
                feats = batch["feats"].to(args.device)
                batch_pred, batch_embed = classifier(feats, return_embed=True)
            elif "image_feats" in batch and "text_feats" in batch:
                # Multi-modal feature mode
                image_feats = batch["image_feats"].to(args.device)
                text_feats = batch["text_feats"].to(args.device)
                batch_pred, batch_embed = classifier(image_feats, text_feats, return_embed=True)
            else:
                raise ValueError("Unexpected batch structure. Supported keys: 'feats' or ('image_feats', 'text_feats')")


            # Detach and move to CPU
            predicted.append(batch_pred.detach().cpu())
            embed.append(batch_embed.detach().cpu())

        # Concatenate all accumulated tensors
        labels = torch.cat(labels, dim=0)
        predicted = torch.cat(predicted, dim=0)
        embed = torch.cat(embed, dim=0)

    return ids, labels, predicted, embed


def eval_metrics(args, labels, predicted, name="dev_seen", epoch=0, compute_loss=True, print_score=True):
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    
    preds_proxy = torch.sigmoid(predicted)
    preds = (preds_proxy >= 0.5).long()
    
    if args.dataset == "MMHS-FineGrained":
        ACCURACY = torchmetrics.Accuracy()
        #TODO
    
    elif args.dataset != "Propaganda":
        ACCURACY = torchmetrics.Accuracy(task='binary')
        AUROC = torchmetrics.AUROC(task='binary')
        PRECISION = torchmetrics.Precision(task='binary')
        RECALL = torchmetrics.Recall(task='binary')
        F1Score = torchmetrics.F1Score(task='binary')
    else:
        ACCURACY = torchmetrics.Accuracy(task="multilabel", num_labels=22, average='micro')
        AUROC = torchmetrics.AUROC(task="multilabel", num_labels=22, average='micro')
        PRECISION = torchmetrics.Precision(task="multilabel", num_labels=22, average='micro')
        RECALL = torchmetrics.Recall(task="multilabel", num_labels=22, average='micro')
        F1Score = torchmetrics.F1Score(task="multilabel", num_labels=22, average='micro')
    acc = ACCURACY(preds, labels)
    roc = AUROC(preds_proxy, labels)
    pre = PRECISION(preds, labels)
    recall = RECALL(preds, labels)
    f1 = F1Score(preds, labels)
    
    if compute_loss:
        lossFn_classifier = nn.BCEWithLogitsLoss()
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        loss = lossFn_classifier(predicted, labels.float())

        if print_score:
            print("{}  Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f} loss: {:.4f} ".format(name, epoch, acc, roc, pre, recall, f1, loss.item()))    
        return acc, roc, pre, recall, f1, loss
    else:
        if print_score:
            print("{} Epoch {} acc: {:.4f} roc: {:.4f} pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(name, epoch, acc, roc, pre, recall, f1))  
        return acc, roc, pre, recall, f1, None

def save_to_json_wandb(args, artifact, name, ids, labels, predicted):
    logging_columns = [
        "id",
        "ground_truth",
        "image",
        "predicted_prob",
        "predicted_label",
    ]
    logging_table = wandb.Table(columns=logging_columns)
    with open(
        "{}/{}.json".format(
            args.output_path,
            name,
        ),
        "w",
    ) as f:
        for i in range(len(ids)):
            id = ids[i]
            label = labels[i].item() if args.dataset != "Propaganda" else labels[i].tolist()
            prob = predicted[i].item() if args.dataset != "Propaganda" else predicted[i].tolist()
            if args.dataset != "Propaganda":
                predicted_label = round(torch.sigmoid(predicted[i]).item())
            else:
                preds_proxy = torch.sigmoid(predicted[i])
                predicted_label = (preds_proxy >= 0.5).long().tolist()
            logging_table.add_data(
                id,
                label,
                artifact.get("{}.png".format(ids[i])),
                prob,
                predicted_label,
            )
            json.dump(
                {
                    "id": id,
                    "gt_label": label,
                    "prob": prob,
                    "predicted_label": predicted_label,
                }, f
            )
            f.write("\n")
            
    # Also save the misclassified ids
    if args.dataset != "Propaganda":
        with open(
            "{}/{}_wrong.json".format(
                args.output_path,
                name,
            ),
            "w",
        ) as f:
            for i in range(len(ids)):
                id = ids[i]
                label = labels[i].item() if args.dataset != "Propaganda" else labels[i].tolist()
                prob = predicted[i].item() if args.dataset != "Propaganda" else predicted[i].tolist()
                predicted_label = round(torch.sigmoid(predicted[i]).item())
                if label != predicted_label:
                    json.dump(
                        {
                            "id": id,
                            "gt_label": label,
                            "prob": prob,
                            "predicted_label": predicted_label,
                        }, f
                    )
                    f.write("\n")
    # Save the table to wandb
    wandb.log({"logging_table_{}".format(name): logging_table})
    
def eval_and_save_epoch_end(args, artifact, train_dl, dev_dl, test_dl, classifier, epoch, compute_loss=True, last_epoch=False):
    classifier.eval()
    pickle_dict = EasyDict()
    epoch_name = "epoch_"+str(epoch)
    pickle_save_path = os.path.join(args.output_path, epoch_name+"_.pkl")
        
    # Dev set
    dev_ids, dev_labels, dev_predicted, dev_embed = iterate_dl(args, dev_dl, classifier)
         
        
    # Train set
    # We do not need the predicted for the train set
    
    
    dev_acc, dev_roc, dev_pre, dev_recall, dev_f1, loss = eval_metrics(args, dev_labels, dev_predicted, name="dev", epoch=epoch, compute_loss=compute_loss)
    test_ids, test_labels, test_predicted, _ = iterate_dl(args, test_dl, classifier)
    test_acc, test_roc, test_pre, test_recall, test_f1, _ = eval_metrics(args, test_labels, test_predicted, name="test", epoch=epoch, compute_loss=False) 
    
    # Try to get args, if not, skip 
    if getattr(args, "save_embed", True):
        train_ids, train_labels, _, train_embed = iterate_dl(args, train_dl, classifier)
        pickle_dict["train_feats"] = train_embed.numpy().astype("float32")
        pickle_dict["train_labels"] = train_labels.numpy().astype("int")
        pickle_dict["train_ids"] = train_ids    
        pickle_dict["evaluate_feats"] = dev_embed.numpy().astype("float32")
        pickle_dict["evaluate_labels"] = dev_labels.numpy().astype("int")
        pickle_dict["evaluate_ids"] = dev_ids 
        with open(pickle_save_path, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #print("pickle of evaluation results saved to: ", pickle_save_path)   
          
    
    if last_epoch and args.save_embed:
        # If the last epoch flag is true
        # Save the prediction json file and wandb table
        save_to_json_wandb(args, artifact,  "dev", dev_ids, dev_labels, dev_predicted)
        save_to_json_wandb(args, artifact, "test", test_ids, test_labels, test_predicted)
        
    return (dev_acc, dev_roc, dev_pre, dev_recall, dev_f1, loss), (test_acc, test_roc, test_pre, test_recall, test_f1)  



