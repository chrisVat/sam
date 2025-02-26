import torch
import numpy as np
import os
import matplotlib.pyplot as plt

SAM_CONFIG = "res/fsam-0.05-pythia-70m-deduped-100-oneshot-130k_mathinstruct_phi-2_3epochs_512_4checkpoints/output/"
DEFAULT_CONFIG = "res/default-pythia-70m-deduped-100-oneshot-130k_mathinstruct_phi-2_3epochs_512_final/output/"

OUTPUT_DIR = "np_cache_analysis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_checkpoint_list(model_path):
    checkpoint_list = []
    for checkpoint in os.listdir(model_path):
        if "checkpoint" in checkpoint:
            new_folder = os.path.join(model_path, checkpoint)
            if os.path.exists(os.path.join(new_folder, "losses.pt")):
                checkpoint_list.append(int(checkpoint.split("-")[1]))
    return checkpoint_list


def boxplot_losses(sam_losses, default_losses, steps):
    for i, step in enumerate(steps):
        cur_sam = sam_losses[i]
        cur_default = default_losses[i]

        # boxplot
        plt.boxplot([cur_sam, cur_default], labels=["SAM", "Default"])
        plt.title(f"Checkpoint {step}")
        plt.ylabel("Loss")
        plt.savefig(f"{OUTPUT_DIR}/checkpoint_{step}.png")
        plt.close()

def plot_full_percentiles(sam_losses, default_losses, steps):
    for i, step in enumerate(steps):
        cur_sam = sam_losses[i]
        cur_default = default_losses[i]

        percentiles = [1, 5, 10, 25, 50, 75, 90, 99]

        sam_percentiles = []
        default_percentiles = []
        for percentile in percentiles:
            sam_percentiles.append(np.percentile(cur_sam, percentile))
            default_percentiles.append(np.percentile(cur_default, percentile))

        #plot it
        plt.plot(percentiles, sam_percentiles, label="SAM")
        plt.plot(percentiles, default_percentiles, label="AdamW")
        plt.title(f"Checkpoint {step} Loss Percentiles")
        plt.xlabel("Percentile")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/percentiles_checkpoint_{step}.png")
        plt.close()


def plot_small_percentiles(sam_losses, default_losses, steps):
    for i, step in enumerate(steps):
        cur_sam = sam_losses[i]
        cur_default = default_losses[i]

        percentiles = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

        sam_percentiles = []
        default_percentiles = []
        for percentile in percentiles:
            sam_percentiles.append(np.percentile(cur_sam, percentile))
            default_percentiles.append(np.percentile(cur_default, percentile))

        plt.plot(percentiles, sam_percentiles, label="SAM")
        plt.plot(percentiles, default_percentiles, label="AdamW")
        plt.title(f"Checkpoint {step} Loss Percentiles")
        plt.xlabel("Percentile")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/small_percentiles_checkpoint_{step}.png")
        plt.close()


def analyze_losses(sam_losses, default_losses, steps):
    # steps is the common checkpoint numbers
    
    #boxplot_losses(sam_losses, default_losses, steps)
    # plot_full_percentiles(sam_losses, default_losses, steps)
    #plot_small_percentiles(sam_losses, default_losses, steps)
    for i, step in enumerate(steps):
        cur_sam = sam_losses[i]
        cur_default = default_losses[i]

        # scatter plot of cur_sam (x) and cur_default (y)
        #h high res
        max_loss = 14
        inc = 0.5

        max_loss = 2
        inc = 0.25

        #"""
        plt.figure(figsize=(10, 10))
        # bright colours!
        plt.hist2d(cur_sam, cur_default, bins=2000, cmap="plasma")
        # add a y=x line?
        b = 0
        m = 1
        plt.plot(cur_sam, m*cur_sam + b, color="black", label="y=x", alpha=0.5, linewidth=0.5)
        plt.legend()
        plt.colorbar(label='Count')
        plt.title(f"Checkpoint {step} Losses", fontsize=18)
        plt.xlabel("SAM Loss", fontsize=14)
        plt.ylabel("AdamW Loss", fontsize=14)
        plt.xlim(0, max_loss)
        plt.ylim(0, max_loss)
        plt.xticks(np.arange(0, max_loss, inc), rotation=45)
        plt.yticks(np.arange(0, max_loss, inc))
        plt.savefig(f"{OUTPUT_DIR}/hexbin_checkpoint_{step}.png")
        plt.close()
        #"""
        """

        plt.figure(figsize=(10, 10))
        plt.scatter(cur_sam, cur_default, alpha=0.1)
        plt.title(f"Checkpoint {step} Losses")
        plt.xlabel("SAM Loss")
        plt.ylabel("AdamW Loss")
        # set x and y lim to 14
        plt.xlim(0, max_loss)
        plt.ylim(0, max_loss)
        plt.xticks(np.arange(0, max_loss, inc), rotation=45)
        plt.yticks(np.arange(0, max_loss, inc))

        b = 0
        m = 1
        plt.plot(cur_sam, m*cur_sam + b, color="red", label="y=x")
        plt.legend()

        plt.savefig(f"{OUTPUT_DIR}/scatter_checkpoint_{step}.png")
        plt.close()
        """



def main():
    sam_ints = get_checkpoint_list(SAM_CONFIG)
    default_ints = get_checkpoint_list(DEFAULT_CONFIG)
    #print(f"SAM: {sam_ints}")
    #print(f"Default: {default_ints}")
    to_use = list(set(sam_ints) & set(default_ints))
    #print(f"Common: {to_use}")
    sam_losses = []
    default_losses = []
    if not os.path.exists(f"{OUTPUT_DIR}/sam_losses.npy"):
        for i in to_use:
            cur_sam_losses = torch.load(f"{SAM_CONFIG}/checkpoint-{i}/losses.pt")
            cur_sam_losses = [loss.item() for loss in cur_sam_losses]
            sam_losses.append(cur_sam_losses)

            cur_default_losses = torch.load(f"{DEFAULT_CONFIG}/checkpoint-{i}/losses.pt")
            cur_default_losses = [loss.item() for loss in cur_default_losses]
            default_losses.append(cur_default_losses)
        sam_losses = [loss for loss in sam_losses]        
        
        sam_losses = np.array(sam_losses)
        default_losses = np.array(default_losses)
        np.save(f"{OUTPUT_DIR}/sam_losses.npy", sam_losses)
        np.save(f"{OUTPUT_DIR}/default_losses.npy", default_losses)
    else:
        sam_losses = np.load(f"{OUTPUT_DIR}/sam_losses.npy")
        default_losses = np.load(f"{OUTPUT_DIR}/default_losses.npy")

    analyze_losses(sam_losses, default_losses, to_use)


if __name__ == "__main__":
    main()


