import torch.nn as nn
import torch
from model.evaluate_rac import retrieve_evaluate_RAC_
from model.classifier import Classifier
from model.loss import compute_loss
import argparse
import wandb
from data_loader.rac_dataloader import CLIP2Dataloader, LMM2Dataloader
from data_loader.dataset import load_feats_from_CLIP, load_feats_from_LMM
from utils.metrics import eval_and_save_epoch_end, compute_metrics_retrieval
from tqdm import tqdm
import numpy as np
import os
import debugpy
import json


def parse_args():

    arg_parser = argparse.ArgumentParser()

    # <----------------- Data Configs ----------------->
    arg_parser.add_argument(
        "--path", type=str, default="./data/")
    arg_parser.add_argument(
        "--output_path", type=str, default="./logging/"
    )
    arg_parser.add_argument("--feature_path", type=str, default="")
    arg_parser.add_argument("--dataset", type=str, default="FB")
    # The threshold for the similarity score for RAC
    arg_parser.add_argument("--similarity_threshold", type=float, default=-1.)

    arg_parser.add_argument(
        "--topk", type=int, default=5, help="Retrieve at most k pairs for validation"
    )

    arg_parser.add_argument("--majority_voting", type=str, default="mean",
                            help="Choose the majority voting method, options are mean, arithmetic, geometric, learned")

    # ----------------- Loss Configs -----------------

    # The loss function for the model is a combination of two parts:
    # Metric class and loss class, both need to be specified

    arg_parser.add_argument("--metric", type=str, default="cos",
                            help="Choose the metric for similarity score, options are cos, ip, l2")
    """
    cos: cosine similarity
    ip: inner product
    l2: l2 distance
    if we use a certain type of metric, we will also use the same criterion for dense retrieval
    """

    arg_parser.add_argument("--loss", type=str, default="naive",
                            help="Choose to use which loss function, options are naive, triplet, contrastive")

    arg_parser.add_argument("--triplet_margin", type=float, default=0.1,
                            help="The margin for triplet loss, epsilon")

    arg_parser.add_argument("--norm_feats_loss", action='store_true',
                        default=False,
                            help="Whether to normalize the feature fpr computing loss ")

    # Do sqrt for L2
    arg_parser.add_argument("--l2_sqrt", action='store_true',
                        default=False,
                            help="Whether to do square root for L2 loss ")

    arg_parser.add_argument("--hybrid_loss", action='store_true',
                        default=False,
                            help="Whether to use logistic loss for the model")
    arg_parser.add_argument("--ce_weight", type=float, default=0.5,
                            help="The weight for the cross entropy loss")
    arg_parser.add_argument("--pos_weight_value", type=float, default=None,
                            help="The weight for the positive samples in the cross entropy loss")

    # <----------------- Model Configs ----------------->
    arg_parser.add_argument("--reinit_cls", action='store_true',
                            default=False, help="Reinitialize the classifier layer")
    arg_parser.add_argument("--cls_path", type=str, default="",)
    # These later args are needed only if reinit the classifier layerz
    arg_parser.add_argument('--num_layers', type=int, default=None)

    # MLP dimension for general
    arg_parser.add_argument('--proj_dim', type=int, default=None)


    arg_parser.add_argument('--input_dim', type=int, default=None)

    arg_parser.add_argument('--dropout', type=float, nargs=3,
                            default=None,
                            help="Set drop probabilities for map, fusion, pre_output")

    # ----------------- Training Configs -----------------
    arg_parser.add_argument("--epochs", type=int, default=5)
    # batch size also sets the number of in_batch positive and in_batch negative
    # we can set limit to the size of in_batch samples
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--lr", type=float, default=0.0001)
    arg_parser.add_argument("--weight_decay", type=float, default=0.0001)
    arg_parser.add_argument(
        "--lr_scheduler",
        action='store_true',
        default=False,
        help="Using LR scheduler or not",
    )
    arg_parser.add_argument("--num_workers", type=int, default=24)
    # default set to zero to match the number of in_batch samples

    arg_parser.add_argument("--grad_clip", type=float,
                            default=0.1, help="Gradient clipping")

    # <----------------- Psuedo Gold Positive Configs ----------------->

    arg_parser.add_argument("--no_pseudo_gold_positives", type=int, default=1)

    # <----------------- Hard Negative Configs ----------------->
    # we need to experiment with different settings here:
    # set a limit ot the number of hard negatives to be retrieved
    # set a threshold for the hard negatives,
    # use single threshold or both of the above threhsolding

    arg_parser.add_argument(
        "--in_batch_loss",
        action='store_true',
        default=False,
        help="Using in batch loss for model training",
    ) 
    
    
    arg_parser.add_argument(
        "--hard_negatives_loss",
        action='store_true',
        default=False,
        help="Using hard negative loss for model training",
    )

    arg_parser.add_argument("--no_hard_negatives", type=int, default=1)
    arg_parser.add_argument("--no_hard_positives", type=int, default=0)

    arg_parser.add_argument(
        "--hard_negatives_multiple",
        type=int,
        default=12,
        help="The value times the no_hard_negatives is the\
                            number of most similar retrieved pairs hard negatives to be retrieved for each sample",
    )
    arg_parser.add_argument(
        "--Faiss_GPU", action='store_true',
                        default=False,
        help="Whether to use GPU for Faiss")

    arg_parser.add_argument(
        "--reindex_every_step",
        action='store_true',
                        default=False,
        help="Whether to reindex the faiss index every step for dense retrieval")

    # For sparse hard negative, not supported for LMM
    # If the sparse dictionary file is not None, we will use sparse retrieval,
    # otherwise, dense retrieval is used as default when the dictioary file is None
    arg_parser.add_argument(
        "--sparse_dictionary",
        type=str,
        default=None,
        help="The name of the file of the sparse retrieval dictionary",
    )
    arg_parser.add_argument(
        "--use_attribute",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Whether to use attribute for object detection in sparse data",
    )
    arg_parser.add_argument(
        "--sparse_topk",
        type=int,
        default=None,
        help="The number of topk retrieved samples for sparse retrieval",
    )
    
    arg_parser.add_argument(
        "--eval_retrieval",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Using retrieval evaluation",
    )

    # <----------------- Logging Configs ----------------->
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument(
        "--final_eval",
        action='store_true',
        default=False,
        help="Doing the final eval or not",
    )

    arg_parser.add_argument("--exp_comment", type=str, default="",
                            help="Optional comment for the experiment")

    arg_parser.add_argument("--group_name", type=str, default="",
                            help=" Name for the wandb group")

    arg_parser.add_argument("--seed", type=int, default=0)

    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--visualise_embed", type=bool, default=False)
    arg_parser.add_argument("--force", action='store_true',
                            default=False, help="Whether to force the run or not")
    arg_parser.add_argument(
        "--save_embed",
        action='store_true',
        default=False,
        help="Save the embedding or not",
    )
    arg_parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Debugging mode",
    )
    
    args = arg_parser.parse_args()

    return args


def model_pass(
    train_dl,
    evaluate_dl,
    test_seen_dl,
    model,
    epochs=0,
    log_interval=10,
    args=None,
    artifacts=None,
    train_set=None,
    sparse_dict=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_dl)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,  # Maximum number of iterations.
            eta_min=1e-5,
        )  # Minimum learning rate.
    global_step = -1

    # Best model criterion
    best_acc = 0.0
    best_roc = 0.0
    best_epoch_path = None
    
    # Before start of training, evaluate the model in both mode
    eval_and_save_epoch_end(
                args, artifacts, train_dl, evaluate_dl, test_seen_dl, model, 0)
    # Retrieval eval
    logging_dict, evaluate_labels = retrieve_evaluate_RAC_(
        train_dl,
        evaluate_dl,
        model,
        largest_retrieval=args.topk,
        threshold=args.similarity_threshold,
        args=args,
        eval_name="dev",
        epoch=0,
    )
    
    acc, roc, pre, recall, f1, prediction, labels = compute_metrics_retrieval(
        logging_dict, evaluate_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True)
        
    logging_dict, test_labels = retrieve_evaluate_RAC_(
        train_dl,
        test_seen_dl,
        model,
        largest_retrieval=args.topk,
        threshold=args.similarity_threshold,
        args=args,
        eval_name="test",
        epoch=0,
    )
    
    acc_test, roc_test, pre_test, recall_test, f1_test, prediction, labels = compute_metrics_retrieval(
        logging_dict, test_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True)
        
        
    
        # Print out the summary of the epoch
    print(
        "Val_Retrieval Epoch  {} acc: {:.4f} roc: {:.4f} \
pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
            0,
            acc,
            roc,
            pre,
            recall,
            f1,
        )
    )
    print(
        "Test_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} \
pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
            0,
            acc_test,
            roc_test,
            pre_test,
            recall_test,
            f1_test,
        )
    )
    
    
    for epoch in tqdm(range(epochs)):
        # train_feats, train_labels is used for dense retrieval for
        # hard negatives and pseudo gold positives
        # When we pass in none, we will force the system
        # to reindex the dense vector embeddings
        # After every epoch, we reindex the dense vector embeddings
        train_feats = None
        train_labels = None
        for step, batch in enumerate(train_dl):
            # Reindex the dense vector embeddings,
            # If we force reindex every step or if it is the first 3 epochs
            if args.reindex_every_step:
                #print("Reindex every step")
                train_feats = None
                train_labels = None
            
            global_step += 1
            (
                total_loss,
                in_batch_loss,
                hard_loss,
                pseudo_gold_loss,
                cross_entropy,
                train_feats,
                train_labels,
            ) = compute_loss(
                batch,
                train_dl,
                model,
                args,
                train_set=train_set,
                sparse_retrieval_dictionary=sparse_dict,
                train_feats=train_feats,
                train_labels=train_labels,
            )
            """if args.sparse_dictionary is None and (args.no_hard_negatives != 0 and args.no_pseudo_gold_positives != 0):
                # Only for dense retrieval
                train_feats = train_feats.detach()
                train_labels = train_labels.detach()"""
            if not((args.no_hard_negatives == 0 and args.no_pseudo_gold_positives == 0) or args.sparse_dictionary is not None): 
                train_feats = train_feats.detach()
                train_labels = train_labels.detach()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()
            if step % log_interval == 0:

                # acc, roc, pre, recall, f1 = compute_metrics_retrieval_baseline(logging_dict, evaluate_labels)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        step,
                        len(train_dl),
                        100.0 * step / len(train_dl),
                        total_loss.item(),
                    )
                )
                if args.loss != "contrastive":
                    hard_loss_val = hard_loss.item() if args.hard_negatives_loss else 0
                    in_batch_loss_val = in_batch_loss.item() if type(
                        in_batch_loss) != int else in_batch_loss
                    pseudo_gold_loss_val = pseudo_gold_loss.item(
                    ) if args.no_pseudo_gold_positives != 0 else 0
                else:
                    # For contrastive loss, we do not have hard negative loss, we only have total loss
                    hard_loss_val = torch.mean(hard_loss).item(
                    ) if args.hard_negatives_loss else 0
                    in_batch_loss_val = torch.mean(in_batch_loss).item() if type(
                        in_batch_loss) != int else in_batch_loss
                    pseudo_gold_loss_val = torch.mean(pseudo_gold_loss).item(
                    ) if args.no_pseudo_gold_positives != 0 else 0
                wandb.log(
                    {
                        "Train/loss": total_loss.item(),
                        "Train/in_batch_loss": in_batch_loss_val,
                        # placeholder for hard negative mining loss
                        "Train/pseudo_gold_loss": pseudo_gold_loss_val,
                        "Train/hard_negative_loss": hard_loss_val,
                        "Train/cross_entropy": cross_entropy,
                        "Train/Lr": optimizer.param_groups[-1]['lr'],
                    },
                    global_step,
                )

        if args.eval_retrieval:
            logging_dict, evaluate_labels = retrieve_evaluate_RAC_(
                train_dl,
                evaluate_dl,
                model,
                largest_retrieval=args.topk,
                threshold=args.similarity_threshold,
                args=args,
                eval_name="dev",
                epoch=epoch,
            )

            acc, roc, pre, recall, f1, prediction, labels = compute_metrics_retrieval(
                logging_dict, evaluate_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True
            )

            logging_dict_test, test_labels = retrieve_evaluate_RAC_(
                train_dl,
                test_seen_dl,
                model,
                largest_retrieval=args.topk,
                threshold=args.similarity_threshold,
                args=args,
                eval_name="test",
                epoch=epoch,
            )

            acc_test, roc_test, pre_test, recall_test, f1_test, prediction, labels = compute_metrics_retrieval(
                logging_dict_test, test_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True
            )
        else:
            acc, roc, pre, recall, f1 = 0, 0, 0, 0, 0
            acc_test, roc_test, pre_test, recall_test, f1_test = 0, 0, 0, 0, 0
        
        wandb.log(
            {
                "Train_Epoch/in_batch_loss": in_batch_loss_val,
                "Train_Epoch/hard_negative_loss": hard_loss_val,
                "Train_Epoch/pseudo_gold_loss": pseudo_gold_loss_val,
                "Train_Epoch/cross_entropy": cross_entropy,

                "Val_Retrieval/accuracy": acc,
                "Val_Retrieval/roc": roc,
                "Val_Retrieval/precision": pre,
                "Val_Retrieval/recall": recall,
                "Val_Retrieval/f1": f1,
                
                "Test_Retrieval/accuracy": acc_test,
                "Test_Retrieval/roc": roc_test,
                "Test_Retrieval/precision": pre_test,
                "Test_Retrieval/recall": recall_test,
                "Test_Retrieval/f1": f1_test


            },
            global_step,
        )

        if args.hybrid_loss:
            # logging at the end of each epoch
            (acc_, roc_, pre_, recall_, f1_, eval_loss_), (acc_test, roc_test, pre_test, recall_test, f1_test) = eval_and_save_epoch_end(
    args, artifacts, train_dl, evaluate_dl, test_seen_dl, model, epoch
)

            
            wandb.log(
                {
                "Val_Probe/acc": acc_,
                "Val_Probe/roc": roc_,
                "Val_Probe/precision": pre_,
                "Val_Probe/recall": recall_,
                "Val_Probe/f1": f1_,
                "Val_Probe/loss": eval_loss_,
                "Test_Probe/acc": acc_test,
                "Test_Probe/roc": roc_test,
                "Test_Probe/precision": pre_test,
                "Test_Probe/recall": recall_test,
                "Test_Probe/f1": f1_test,
                    
                },
                global_step,
            )


            
        # Print out the summary of the epoch
        print(
            "Val_Retrieval Epoch  {} acc: {:.4f} roc: {:.4f} \
pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                epoch,
                acc,
                roc,
                pre,
                recall,
                f1,
            )
        )
        print(
            "Test_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} \
pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                epoch,
                acc_test,
                roc_test,
                pre_test,
                recall_test,
                f1_test,
            )
        )
        # print new line
        print(" ")
        # Save the model if the val criterion is the best so far
        acc_ = acc_ if args.hybrid_loss else acc
        if acc_ > best_acc:
            print("Current Epoch Acc: ", acc_, "Best model so far, saving...")
            best_acc = acc_

            # Delete the previous best model
            #if best_epoch_path is not None:
            #    if os.path.exists(best_epoch_path):
            #        os.remove(best_epoch_path)

            best_epoch_path = args.output_path + \
                "/ckpt/best_model_{}_{}.pt".format(epoch, str(acc_))

            torch.save(
                model.state_dict(),
                best_epoch_path
            )
        # If last epoch or early stop, save the model
        if epoch == args.epochs - 1:
            print("Last Epoch, saving...")
            torch.save(
                model.state_dict(),
                args.output_path +
                "/ckpt/last_model_{}_{}.pt".format(epoch, acc)
            )
    return model, best_epoch_path


def main(args):

    # <----------------- Name the model ----------------->

    if args.metric == "cos":
        loss_str = "cosSim"
    elif args.metric == "ip":
        loss_str = "innerProduct"
    elif args.metric == "l2":
        loss_str = "L2"

    if args.loss == "naive":
        loss_str += "_naive"
    elif args.loss == "triplet":
        loss_str += "_triplet"
    elif args.loss == "contrastive":
        loss_str += "_contrastive"

    hard_negative_name = "_hard_negative_{}".format(
        args.no_hard_negatives)
    
    if args.no_pseudo_gold_positives!=0 and args.no_hard_positives !=0:
        positive_name = "_PseudoGold_positive_{}_hard_positive_{}".format(
            args.no_pseudo_gold_positives, args.no_hard_positives)
    elif args.no_pseudo_gold_positives!=0:
        positive_name = "_PseudoGold_positive_{}".format(
            args.no_pseudo_gold_positives)
    elif args.no_hard_positives !=0:
        positive_name = "_hard_positive_{}".format(
            args.no_hard_positives)
    else:
        positive_name = "inbatch_positive"

    # Get date and time
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if "llava-v1.5-7b" in args.feature_path:
        model_name = "llava-v1.5-7b" 
    elif "qwen2vl7b" in args.feature_path:
        model_name = "qwen2vl7b"
    elif "qwen2vl2b" in args.feature_path:
        model_name = "qwen2vl2b"
    else:
        AssertionError("Model name not found")
    
    group_name = date + args.feature_path.split("/")[-1].replace("train_", "").replace(".pt", "")
    exp_name = "RAC_{}_lr{}_Bz{}_Ep{}_{}_seed{}{}{}{}".format(
        model_name,
        args.lr,
        args.batch_size,
        args.epochs,
        loss_str,
        args.seed,
        "_hybrid_loss" if args.hybrid_loss else "",
        args.exp_comment,
        "_{}".format(args.sparse_dictionary) if args.sparse_dictionary is not None else "",
    )
    # Construct output path
    args.output_path = os.path.join(
        args.output_path, "Retrieval", args.dataset, args.group_name, exp_name, "")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + "/ckpt/")
    else:
        if not args.force:
            print(args.force)
            # Abort avoid overwriting
            raise Exception("Output path already exists, aborting...")

    
    print(args)
    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached")
    else:
        run = wandb.init(
            entity="jingbiao",
            project="MMHS-Retrieval",
            name=exp_name,
            config={
                "dataset": args.dataset,
                "learning_rate": args.lr,
                "lr_scheduler": args.lr_scheduler,
                "weight_decay": args.weight_decay,
                "Dropout": args.dropout,
                "Gradient clipping": args.grad_clip,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "Similarity minimum threshold": args.similarity_threshold,
                "Maximum top-K retrieved for Classification": args.topk,
                "Model": args.feature_path,
                "Loss": loss_str,
                "Hard negative loss": args.hard_negatives_loss,
                "Hard negative samples to retrieve": args.no_hard_negatives,
                "Pseudo gold positives": args.no_pseudo_gold_positives,
                "Experiment Comment": args.exp_comment,
                "Majority_voting": args.majority_voting,
                "Sparse dictionary": args.sparse_dictionary,
                "Use in batch samples": args.in_batch_loss,
            },
            group=group_name,
        )

    # <----------------- Load the data ----------------->

    train, dev, test_seen = load_feats_from_LMM(
        args.feature_path, args.dataset
        )

    train_dl, dev_dl, test_seen_dl = LMM2Dataloader(
        train,
        dev,
        test_seen,
        batch_size=args.batch_size,
    )


    # Load the sparse retrieval dictionary if the path is not None
    if args.sparse_dictionary is not None:
        sparse_dict = {}
        for line in open(
            os.path.join(
                args.path,
                "Sparse_Retrieval_Dict",
                args.dataset,
                args.sparse_dictionary+".json"
            ), "r"
        ):
            subdict = json.loads(line)
            sparse_dict[subdict["id"]] = subdict
    else:
        sparse_dict = None

    # <----------------- Construct the model ----------------->
   
    #list(enumerate(train_dl))

    #model = classifier_hateClipper(
    #    image_feat_dim, text_feat_dim, args.num_layers, args.proj_dim,
    #    args.map_dim, args.fusion_mode, dropout=args.dropout, batch_norm=args.batch_norm, args=args)
    
    classifier_config_path = os.path.join(args.cls_path, 'classifier_config.json')
    if not args.reinit_cls:
        classifier_bin_path = os.path.join(args.cls_path, 'classifier.bin')
        with open(classifier_config_path, "r") as f:
            classifier_config = json.load(f)

        model = Classifier(
            input_shape=classifier_config["hidden_size"],
            num_layers=classifier_config["num_layers"],
            proj_dim=classifier_config["proj_dim"],
            output_dim=classifier_config["output_dim"],
            input_dropout=classifier_config["input_dropout"],
            dropout=classifier_config["dropout"]
        )  
        args.proj_dim = classifier_config["proj_dim"]
    else:
        input_shape = args.input_dim
        
        model = Classifier(
            input_shape=input_shape,
            num_layers=args.num_layers,
            proj_dim=args.proj_dim,
            output_dim=1,
            input_dropout=0,
            dropout=args.dropout
        )
    
    model.to(args.device)
    print(model)
    #evaluate_split(train, dev, "dev")
    # Load checkpoint
    if not args.reinit_cls:
        model.load_state_dict(torch.load(classifier_bin_path))
    
    # <----------------- Train the model ----------------->
    model, best_epoch_path = model_pass(
        train_dl,
        dev_dl,
        test_seen_dl,
        model,
        epochs=args.epochs,
        log_interval=args.log_interval,
        args=args,
        artifacts=None,
    )
    run.finish()

if __name__ == "__main__":
    args = parse_args()

    # set the seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
