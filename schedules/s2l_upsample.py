import torch, numpy as np, glob, os, sys
sys.path.insert(0, os.path.abspath('.'))
from schedule_base import Schedule


class S2LUpsample(Schedule):
    def __init__(self, model, tokenizer, args):
        super(S2LUpsample, self).__init__(model, tokenizer, args)

        self.upsample_amount = args.get("upsample_amount", 3) # increase in example usage 
        self.num_cluster = args.get("num_cluster", 2) # of clusters per source, highest value loss is upsampled per source

        self.sources = np.array([d["source"] for d in self.train_data])

        if torch.distributed.get_rank() != 0:
            return

        losses = []

        for ck in glob.glob(f'{args["ref_model_path"]}/*'):
            try:
                losses.append(torch.tensor(torch.load(os.path.join(ck, "losses.pt"))))
            except Exception as e:
                print(f"Could not load {ck}/losses.pt: {e}")
        
        print("self.losses:", len(losses), "losses loaded from checkpoints")        
        if not losses:
            self.losses = torch.zeros(self.n_pool, 1)
        else:
            self.losses = torch.stack(losses, 1).float()
            self.losses[torch.isnan(self.losses)] = 0

        self.losses   = self.losses[self.train_idx]
        self.loss_vec = self.losses.mean(1)

    # ------------------------------------------------------------------
    def initialize_labeled_data(self):
        if torch.distributed.get_rank() != 0:
            return
        print("initializing labeled data...")

        num = self.init_label_num
        sources, counts = np.unique(self.sources, return_counts=True)
        sorted_source_idx = np.argsort(counts)

        average_example_per_source = self.n_pool / len(sorted_source_idx)

        sampled = []

        # how many times each example is shown
        per_example_usages = np.zeros(self.n_pool, dtype=int)

        for i in range(len(sorted_source_idx)):
            print("source", i, "of", len(sorted_source_idx), ":", sources[sorted_source_idx[i]], counts[sorted_source_idx[i]])
            # check number of examples in this source
            num_source_examples = counts[sorted_source_idx[i]]
            
            indices = np.where(self.sources == sources[sorted_source_idx[i]])[0]
            per_example_usages[indices] = 1 

            if num_source_examples < average_example_per_source:
                # upsample all per_example_usages to upsample_amount + 1
                # get all indices of the source examples, and set the per example usage of those indices to num_source_examples + 1
                per_example_usages[indices] = int(self.upsample_amount + 1)
            else:
                new_indices = self.faiss_kmeans_selection(self.losses[indices], self.num_cluster)
                per_example_usages[new_indices] = int(self.upsample_amount + 1)

        self.per_example_usages = per_example_usages

        print(f"Built per-example usage list: {self.per_example_usages.sum()} total samples for next epoch")
        print("Per-example usages:", self.per_example_usages)
        print("running debug summary...")
        self.debug_summary()
        print("done running debug summary.")


    def faiss_kmeans_selection(self, losses: np.ndarray, num_cluster: int = 2):
        # losses: shape (N, T) — N examples, T loss checkpoints
        assert losses.ndim == 2, "Expected losses of shape (num_examples, num_checkpoints)"
        
        x = losses.astype(np.float32)

        # Run k-means clustering on loss trajectories
        kmeans = faiss.Kmeans(d=x.shape[1], k=num_cluster, niter=20, verbose=False)
        kmeans.train(x)
        _, cluster_ids = kmeans.index.search(x, 1)
        cluster_ids = cluster_ids.squeeze()  

        avg_loss_per_cluster = []
        for k in range(num_cluster):
            mask = (cluster_ids == k)
            avg_loss = losses[mask].mean() if np.any(mask) else -np.inf
            avg_loss_per_cluster.append(avg_loss)

        target_cluster = int(np.argmax(avg_loss_per_cluster))
        selected_indices = np.where(cluster_ids == target_cluster)[0]
        return selected_indices

    def debug_summary(self):
        import matplotlib.pyplot as plt
        import os

        if self.per_example_usages.sum() == 0:
            print("=== S2LUpsample SUMMARY ===")
            print("No usages recorded.")
            print("================================")
            return

        used_mask = self.per_example_usages > 0
        total_per_source = {s: int((self.sources == s).sum()) for s in sorted(set(self.sources))}
        unique_kept = {s: int(((self.sources == s) & used_mask).sum()) for s in total_per_source}
        total_usages = {s: int(self.per_example_usages[self.sources == s].sum()) for s in total_per_source}

        print("=== S2LUpsample SUMMARY ===")
        print(f"Total examples       : {self.n_pool}")
        print(f"Used this epoch      : {self.per_example_usages.sum()} total instances")
        print(f"Unique used          : {used_mask.sum()} examples")

        print("--- per source ---")
        print(f"{'SOURCE':<45} | {'unique_used / total':<20} | {'shown total'}")
        print("-" * 90)
        for s in sorted(total_per_source):
            print(f"{s:<45} | {unique_kept[s]:>4} / {total_per_source[s]:<14} | {total_usages[s]:>5}")

        lv = self.loss_vec[used_mask]
        print("--- difficulty stats (used) ---")
        print(f"  min  {lv.min():.4f}")
        print(f"  mean {lv.mean():.4f}")
        print(f"  max  {lv.max():.4f}")
        print(f"  std  {lv.std():.4f}")
        print("================================")

        # Save histogram
        save_dir = "usage_hist/"
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(8, 4))

        # Create bins for every integer value from 0 to max
        max_val = int(np.max(self.per_example_usages))
        bins = np.arange(0, max_val + 2)  # +2 to include the last integer

        plt.hist(self.per_example_usages, bins=bins, edgecolor='black')

        plt.title("Per-Example Usages Histogram")
        plt.xlabel("Usage Count")
        plt.ylabel("Number of Examples")
        plt.xticks(np.arange(0, max_val + 1, 1))  # Set x-axis ticks at integer values
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"{self.alpha}__{self.beta}__{self.min_w}__{self.max_unique}.png"))
        plt.close()
