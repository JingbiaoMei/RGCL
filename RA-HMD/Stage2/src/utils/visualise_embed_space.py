import numpy as np
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from matplotlib.transforms import Bbox
import imageio.v2 as imageio
import wandb
from sklearn.manifold import TSNE

def gen_name(epoch, rac=True):
    if rac:
        return "devepoch_"+str(epoch)+"_retrieval_logging_dict.pkl"
    else:
        return "devepoch_"+str(epoch)+"_.pkl"


def normalise(x):
    return x/np.linalg.norm(x, axis=1, keepdims=True)


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def plot_embedding_pca(data_path=None, save_path=None, epoch=4, name="TopTitle", normalised_from_dict=False, save=True, fps=5, log2wandb=False):
    # Plot the embedding space evolution for first 4 epochs for default
    # plt.rcParams['figure.figsize'] = [8*epoch/2, 8]
    save_path = os.path.join(data_path, "pca") if save_path is None else save_path
    # For calling as a function
    if log2wandb:
        os.mkdir(save_path)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(2, epoch, figsize=(
        16*epoch/2, 16), sharex=False, sharey=False)
    fig.suptitle(name)
    # Defining custom 'xlim' and 'ylim' values.
    custom_xlim = (-1, 1)
    custom_ylim = (-1, 1)

    # Setting the values for all axes.
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)

    for i in tqdm(range(epoch), desc="Plotting Embedding Space With PCA"):
        # Here we add robustness to the code,

        # if the epoch is not found, we will skip it

        data_path+"/"+gen_name(i)
        
        try:
            with open(data_path+"/"+gen_name(i), "rb") as f:
                pickle_dict = pickle.load(f)
                if normalised_from_dict:
                    train_feats = pickle_dict["train_feats_normalized"]
                    evaluate_feats = pickle_dict["evaluate_feats_normalized"]
                else:
                    train_feats = pickle_dict["train_feats"]
                    evaluate_feats = pickle_dict["evaluate_feats"]
                train_labels = pickle_dict["train_labels"]
                train_ids = pickle_dict["train_ids"]
                evaluate_labels = pickle_dict["evaluate_labels"]
                evaluate_ids = pickle_dict["evaluate_ids"]
        except:
            print("Path:{}, Epoch: {} not found".format(data_path+"/"+gen_name(i), i))

            # break the loop
            break

        if not normalised_from_dict:
            train_feats = normalise(train_feats)
            evaluate_feats = normalise(evaluate_feats)

        # Train
        U, S, V = np.linalg.svd(train_feats)
        X1 = np.matmul(train_feats, V[0])
        X2 = np.matmul(train_feats, V[1])
        # Plot
        # plt.subplot(2, epoch, i+1)
        # plt.scatter(X1, X2, c=train_labels)
        # plt.title("Train Epoch "+str(i))
        axs[0, i].scatter(X1, X2, c=train_labels)
        axs[0, i].set_title("Train Epoch "+str(i))

        # Evaluate
        U, S, V = np.linalg.svd(evaluate_feats)
        X1 = np.matmul(evaluate_feats, V[0])
        X2 = np.matmul(evaluate_feats, V[1])
        # plt.subplot(2, epoch, i+epoch+1)
        # plt.scatter(X1, X2, c=evaluate_labels)
        # plt.title("Val Epoch "+str(i))
        axs[1, i].scatter(X1, X2, c=evaluate_labels)
        axs[1, i].set_title("Val Epoch "+str(i))

    if save:
        # Save the full plot
        fig.savefig(save_path+"/"+"embed_space_Full.png",
                    bbox_inches="tight", dpi=300)
        # sub plots are saved as individual files, used to create gif
        # Initialise the list of filenames for gif
        filenames_train = []
        filenames_val = []

        for i in range(epoch):
            # Train
            extent = full_extent(axs[0, i]).transformed(
                fig.dpi_scale_trans.inverted())
            filename = save_path+"/" + \
                "embed_space_train_epoch_{}.png".format(i)
            filenames_train.append(filename)
            fig.savefig(filename, bbox_inches=extent, dpi=300)
            # Evaluate
            extent = full_extent(axs[1, i]).transformed(
                fig.dpi_scale_trans.inverted())
            filename = save_path+"/" + \
                "embed_space_val_epoch_{}.png".format(i)
            filenames_val.append(filename)
            fig.savefig(filename, bbox_inches=extent, dpi=300)

    images = []
    for img in filenames_train:
        images.append(imageio.imread(img))
        images.append(imageio.imread(img))
    
    imageio.mimwrite(
        save_path+"/"+"embed_space_train.gif", images, duration=1000/fps)
    images = []
    for img in filenames_val:
        images.append(imageio.imread(img))
        images.append(imageio.imread(img))
    imageio.mimwrite(
        save_path+"/"+"embed_space_val.gif", images, duration=1000/fps)
    """with imageio.get_writer(save_path+"/"+"embed_space_train.gif", mode='I') as writer:
        for filename in filenames_train:
            image = imageio.imread(filename)
            writer.append_data(image)
    with imageio.get_writer(save_path+"/"+"embed_space_val.gif", mode='I') as writer:
        for filename in filenames_val:
            image = imageio.imread(filename)
            writer.append_data(image)"""
    if log2wandb:
        # Log the gif and full image to wandb
        wandb.log({"PCA_embed_space_train_GIF": wandb.Image(save_path+"/"+"embed_space_train.gif")})
        wandb.log({"PCA_embed_space_val_GIF": wandb.Image(save_path+"/"+"embed_space_val.gif")})
        wandb.log({"PCA_embed_space_Full": wandb.Image(save_path+"/"+"embed_space_Full.png")})
    return None


def plot_embedding_tsne(data_path=None, save_path=None,n_components = 2, epoch=4, name="TopTitle", normalised_from_dict=False, save=True, fps=5, log2wandb=False):
    save_path = os.path.join(data_path, "tsne") if save_path is None else save_path
    
    # For calling as a function
    if log2wandb:
        os.mkdir(save_path)
    # Plot the embedding space evolution for first 4 epochs for default
    # plt.rcParams['figure.figsize'] = [8*epoch/2, 8]
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(2, epoch, figsize=(
        16*epoch/2, 16), sharex=False, sharey=False)
    fig.suptitle(name)
    # Defining custom 'xlim' and 'ylim' values.
    #custom_xlim = (-80, 80)
    #custom_ylim = (-80, 80)

    # Setting the values for all axes.
    #plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    min_train = min_val = 100
    max_train = max_val = -100

    for i in tqdm(range(epoch), desc="Plotting Embedding Space With TSNE"):
        # Here we add robustness to the code,

        # if the epoch is not found, we will skip it

        try:
            with open(data_path+"/"+gen_name(i), "rb") as f:
                pickle_dict = pickle.load(f)
                
        except:
            print("Path:{}, Epoch: {} not found".format(data_path+"/"+gen_name(i), i))

            # break the loop
            break
        if normalised_from_dict:
            train_feats = pickle_dict["train_feats_normalized"]
            evaluate_feats = pickle_dict["evaluate_feats_normalized"]
        else:
            train_feats = pickle_dict["train_feats"]
            evaluate_feats = pickle_dict["evaluate_feats"]
        train_labels = pickle_dict["train_labels"]
        train_ids = pickle_dict["train_ids"]
        evaluate_labels = pickle_dict["evaluate_labels"]
        evaluate_ids = pickle_dict["evaluate_ids"]

        # Train

        # Plot
        # plt.subplot(2, epoch, i+1)
        # plt.scatter(X1, X2, c=train_labels)
        # plt.title("Train Epoch "+str(i))
        
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(train_feats)
        axs[0, i].scatter(tsne_result[:,0], tsne_result[:,1], c=train_labels)
        axs[0, i].set_title("Train Epoch "+str(i))
        min_train = tsne_result.min() if tsne_result.min() < min_train else min_train
        max_train = tsne_result.max() if tsne_result.max() > max_train else max_train
        # Evaluate
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(evaluate_feats)
        # plt.subplot(2, epoch, i+epoch+1)
        # plt.scatter(X1, X2, c=evaluate_labels)
        # plt.title("Val Epoch "+str(i))
        axs[1, i].scatter(tsne_result[:,0], tsne_result[:,1], c=evaluate_labels)
        axs[1, i].set_title("Val Epoch "+str(i))
        min_val = tsne_result.min() if tsne_result.min() < min_val else min_val
        max_val = tsne_result.max() if tsne_result.max() > max_val else max_val
    # Set the limits of the plot to the limits of the data
    lim = (min_train, max_train)    
    plt.setp(axs[0, :], xlim=lim, ylim=lim,aspect='equal')
    lim = (min_val, max_val)
    plt.setp(axs[1, :], xlim=lim, ylim=lim,aspect='equal')
    if save:
        # Save the full plot
        fig.savefig(save_path+"/"+"embed_space_Full.png",
                    bbox_inches="tight", dpi=300)
        # sub plots are saved as individual files, used to create gif
        # Initialise the list of filenames for gif
        filenames_train = []
        filenames_val = []

        for i in range(epoch):
            # Train
            extent = full_extent(axs[0, i]).transformed(
                fig.dpi_scale_trans.inverted())
            filename = save_path+"/" + \
                "embed_space_train_epoch_{}.png".format(i)
            filenames_train.append(filename)
            fig.savefig(filename, bbox_inches=extent, dpi=300)
            # Evaluate
            extent = full_extent(axs[1, i]).transformed(
                fig.dpi_scale_trans.inverted())
            filename = save_path+"/" + \
                "embed_space_val_epoch_{}.png".format(i)
            filenames_val.append(filename)
            fig.savefig(filename, bbox_inches=extent, dpi=300)

    images = []
    for img in filenames_train:
        images.append(imageio.imread(img))
        images.append(imageio.imread(img))
    
    imageio.mimwrite(
        save_path+"/"+"embed_space_train.gif", images, duration=1000/fps)
    images = []
    for img in filenames_val:
        images.append(imageio.imread(img))
        images.append(imageio.imread(img))
    imageio.mimwrite(
        save_path+"/"+"embed_space_val.gif", images, duration=1000/fps)
    """with imageio.get_writer(save_path+"/"+"embed_space_train.gif", mode='I') as writer:
        for filename in filenames_train:
            image = imageio.imread(filename)
            writer.append_data(image)
    with imageio.get_writer(save_path+"/"+"embed_space_val.gif", mode='I') as writer:
        for filename in filenames_val:
            image = imageio.imread(filename)
            writer.append_data(image)"""
    if log2wandb:
        # Log the gif and full image to wandb
        wandb.log({"TSNE_embed_space_train_GIF": wandb.Image(save_path+"/"+"embed_space_train.gif")})
        wandb.log({"TSNE_embed_space_val_GIF": wandb.Image(save_path+"/"+"embed_space_val.gif")})
        wandb.log({"TSNE_embed_space_Full": wandb.Image(save_path+"/"+"embed_space_Full.png")})
    return None

def plot_all_under_dir(args):
    for subdir in tqdm(os.listdir(args.data_path), desc="Interate over all Subdir in a Group"):
        print(subdir)
        data_path = args.data_path+"/"+subdir
        # If the image is already generated, we will skip it
        
        if args.pca:
            save_path_pca = os.path.join(data_path, "pca")
            if not os.path.exists(save_path_pca) :
                os.makedirs(save_path_pca, exist_ok=False)
                plot_embedding_pca(data_path=data_path, save_path=save_path_pca, epoch=args.epoch, name=subdir)
            elif args.overwrite:
                plot_embedding_pca(data_path=data_path, save_path=save_path_pca, epoch=args.epoch, name=subdir)
            else:
                print("Skipping, PCA already exists")
                
            
        if args.tsne:
            save_path_tsne = os.path.join(data_path, "tsne")
            if not os.path.exists(save_path_tsne):
                os.makedirs(save_path_tsne, exist_ok=False)
                plot_embedding_tsne(data_path=data_path, save_path=save_path_tsne, epoch=args.epoch, name=subdir)
            elif args.overwrite:
                plot_embedding_tsne(data_path=data_path, save_path=save_path_tsne, epoch=args.epoch, name=subdir)
            else:
                print("Skipping, TSNE already exists")
                

if __name__ == "__main__":
        # Construct argument parser
    parser = argparse.ArgumentParser(description='Visualise Embedding Space')
    parser.add_argument('--data_path', type=str,
                        default="./logging/Retrieval/RAC_PseudoGold_1_11June", help='path to the data')

    parser.add_argument('--pca', type=bool,
                        default=False, help='Using PCA or not')
    parser.add_argument('--tsne', type=bool,
                        default=False, help='Using t-SNE or not')
    parser.add_argument('--wandb', type=bool,
                        default=False, help='Using wandb or not')
    parser.add_argument('--overwrite', type=bool,
                        default=False, help='Overwrite the existing file or not')
    parser.add_argument('--epoch', type=int,
                        default=10, help='Number of epoch to plot')
    args = parser.parse_args()
    print(args)
    plot_all_under_dir(args)
