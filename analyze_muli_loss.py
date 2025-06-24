import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

SAM_CONFIG = "res/preconfsam-0.10-mini-4k-mathinstruct_phi-2_3epochs_900-1gpu_lr2e-5_bs32/output/"
DEFAULT_CONFIG = "res/default-mathinstruct_mini-2epochs_900-1gpu_lr2e-5_bs32/output/"

OUTPUT_DIR = "multi_np_cache_analysis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_checkpoint_list(model_path):
    checkpoint_list = []
    for checkpoint in os.listdir(model_path):
        if "checkpoint" in checkpoint:
            new_folder = os.path.join(model_path, checkpoint)
            if os.path.exists(os.path.join(new_folder, "losses.pt")):
                checkpoint_list.append(int(checkpoint.split("-")[1]))
    checkpoint_list.sort()
    return checkpoint_list

def analyze_trajectories(sam_losses, default_losses, checkpoints, name="sam_default"):
    default_means = np.mean(default_losses, axis=0)
    sam_means = np.mean(sam_losses, axis=0)

    # Color based on percentile of default_means
    colors, default_percentiles = get_percentile_colormap(default_means)

    analyze_trajectories_contrast(sam_losses, default_losses, checkpoints, name + "-contrast", colors)
    analyze_clustered_trajectories(sam_losses, default_losses, checkpoints, name + "-clustered", colors)
    plot_percentile_examples(sam_losses, default_losses, checkpoints, name + "-percentile", default_means)


def get_percentile_colormap(values):
    ranks = np.argsort(np.argsort(values))
    percentiles = ranks / (len(values) - 1)  # normalized 0-1
    colormap = cm.get_cmap("autumn")
    colors = colormap(percentiles)
    return colors, percentiles

def plot_percentile_examples(sam_losses, default_losses, steps, name="percentile_examples", default_means=None):
    num_checkpoints, num_samples = sam_losses.shape

    if default_means is None:
        default_means = np.mean(default_losses, axis=0)

    # Compute percentiles and find closest indices
    percentile_jump = 10
    percentiles = np.percentile(default_means, np.linspace(percentile_jump, 100, 100))
    closest_indices = [np.argmin(np.abs(default_means - p)) for p in percentiles]

    # Normalize based on position in list (i.e., 1st to 100th percentile = 0 to 1)
    normed_percentiles = np.linspace(0, percentile_jump, 100)
    colormap = cm.get_cmap("autumn")

    plt.figure(figsize=(14, 8))
    for i, idx in enumerate(closest_indices):
        color = colormap(normed_percentiles[i])
        plt.plot(steps, sam_losses[:, idx], linestyle='-', color=color, alpha=0.8)
        plt.plot(steps, default_losses[:, idx], linestyle='--', color=color, alpha=0.8)

    plt.xlabel("Checkpoint")
    plt.ylabel("Loss")
    plt.ylim(0, 5.0)
    plt.title("SAM vs Default: 1st to 100th Percentile Trajectories (Dashed = Default, Solid = SAM)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png")



def analyze_trajectories_contrast(sam_losses, default_losses, steps, name="contrast", colors=None, num_samples=50):
    num_checkpoints, total_samples = sam_losses.shape
    mean_diff = np.mean(sam_losses - default_losses, axis=0)

    sorted_indices = np.argsort(mean_diff)
    sampled_indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
    chosen_indices = sorted_indices[sampled_indices]

    plt.figure(figsize=(14, 7))
    for idx in chosen_indices:
        color = colors[idx] if colors is not None else "gray"
        plt.plot(steps, default_losses[:, idx], color=color, alpha=0.4, linestyle="--")
        plt.plot(steps, sam_losses[:, idx], color=color, alpha=0.9)

    plt.xlabel("Checkpoint")
    plt.ylabel("Loss")
    plt.title("SAM vs Default Trajectories (Solid=SAM, Dashed=Default)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png")


def analyze_clustered_trajectories(sam_losses, default_losses, steps, name="clustered_contrast", colors=None, num_clusters=50):
    def cluster_and_get_representatives(losses, num_clusters):
        losses_T = losses.T
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(losses_T)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        representatives = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_data = losses_T[cluster_indices]
            centroid = centroids[i]
            distances = cdist(cluster_data, [centroid])
            closest_idx_in_cluster = cluster_indices[np.argmin(distances)]
            representatives.append(closest_idx_in_cluster)
        return np.array(representatives)

    sam_reps = cluster_and_get_representatives(sam_losses, num_clusters)
    default_reps = cluster_and_get_representatives(default_losses, num_clusters)

    # Plot both SAM and Default on the same axes
    plt.figure(figsize=(14, 8))
    for sam_idx, default_idx in zip(sam_reps, default_reps):
        color = colors[default_idx] if colors is not None else "gray"
        plt.plot(steps, sam_losses[:, sam_idx], color=color, linestyle='-', alpha=0.9)
        plt.plot(steps, default_losses[:, default_idx], color=color, linestyle='--', alpha=0.9)

    plt.xlabel("Checkpoint")
    plt.ylabel("Loss")
    plt.title("Clustered Representative Trajectories (Dashed = Default, Solid = SAM)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png")




def load_double():
    sam_ints = get_checkpoint_list(SAM_CONFIG)
    default_ints = get_checkpoint_list(DEFAULT_CONFIG)
    print(f"SAM: {sam_ints}")
    print(f"Default: {default_ints}")
    to_use = list(set(sam_ints) & set(default_ints))
    to_use = sorted(to_use)

    print(f"Common: {to_use}")
    sam_losses = []
    default_losses = []

    run_anyway=True
    if os.path.exists(f"{OUTPUT_DIR}/sam_losses.npy") and os.path.exists(f"{OUTPUT_DIR}/default_losses.npy"):
        sam_losses = np.load(f"{OUTPUT_DIR}/sam_losses.npy")
        default_losses = np.load(f"{OUTPUT_DIR}/default_losses.npy")
        run_anyway=False
        if sam_losses.shape[0] != len(to_use) or default_losses.shape[0] != len(to_use):
            print("Shape mismatch, reloading...")
            run_anyway = True
            sam_losses = []
            default_losses = []
    if run_anyway:
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



    for step in range(sam_losses.shape[0]):
        print(f"Step {step} SAM: {sam_losses[step].mean():.6f} +/- {sam_losses[step].std():.6f}")
        print(f"Step {step} DEF: {default_losses[step].mean():.6f} +/- {default_losses[step].std():.6f}")

    analyze_trajectories(sam_losses, default_losses, to_use, name="sam_default")


def full_traj_analysis(losses, name="sam"):
    return
    analyze_single_trajectory(losses, range(len(losses)), name=name)
    analyze_trajectories_density(losses, range(len(losses)), name=name + "-density")
    analyze_trajectories_colored(losses, range(len(losses)), name=name + "-colored")


def load_singles():
    sam_losses = load_single(sam=True)
    default_losses = load_single(sam=False)
    full_traj_analysis(sam_losses, name="sam")
    full_traj_analysis(default_losses, name="default")


def load_single(sam=True):
    config = SAM_CONFIG if sam else DEFAULT_CONFIG
    ints = get_checkpoint_list(config)
    print(f"Checkpoint list: {ints}")
    losses = []

    file_name = "all_sam_losses.npy" if sam else "all_default_losses.npy"
    
    run_anyway=True
    if os.path.exists(f"{OUTPUT_DIR}/{file_name}"):
        run_anyway=False
        losses = np.load(f"{OUTPUT_DIR}/{file_name}")
        if losses.shape[0] != len(ints):
            print("Shape mismatch, reloading...")
            run_anyway = True
            losses = []
    if not os.path.exists(f"{OUTPUT_DIR}/{file_name}") or run_anyway:
        for i in ints:
            cur_losses = torch.load(f"{config}/checkpoint-{i}/losses.pt")
            cur_losses = [loss.item() for loss in cur_losses]
            losses.append(cur_losses)
        losses = [loss for loss in losses]        
        losses = np.array(losses)
        np.save(f"{OUTPUT_DIR}/{file_name}", losses)
    return losses


def main():    
    load_singles()
    load_double()


if __name__ == "__main__":
    main()


