import re
import os
import matplotlib.pyplot as plt

# === Path to your loss report file ===
file_path = "loss_reports.txt"

# Create output directory
os.makedirs("dataset_usage", exist_ok=True)

# Read file content
with open(file_path, "r") as f:
    raw_text = f.read()

# Storage
usage_vals = []
alpha_vals = []
beta_vals = []
eval1_vals = []
eval2_vals = []
eval3_vals = []

# Regex pattern to extract alpha, beta, usage, and 3 evals
pattern = (
    r"loss_priority_alpha: ([\d\.]+).*?"
    r"source_balance_beta: ([\d\.]+).*?"
    r"max_unique: .*?"
    r"max_dup_per_example: .*?"
    r"dataset usage: (\d+)%.*?"
    r"Eval 1: \d+ entries, Avg Loss = ([\d\.]+).*?"
    r"Eval 2: \d+ entries, Avg Loss = ([\d\.]+).*?"
    r"Eval 3: \d+ entries, Avg Loss = ([\d\.]+)"
)

for match in re.finditer(pattern, raw_text, re.DOTALL):
    alpha_vals.append(float(match.group(1)))
    beta_vals.append(float(match.group(2)))
    usage_vals.append(int(match.group(3)))
    eval1_vals.append(float(match.group(4)))
    eval2_vals.append(float(match.group(5)))
    eval3_vals.append(float(match.group(6)))

# Baseline losses
baseline_eval1 = 2.8964
baseline_eval2 = 2.4900
baseline_eval3 = 2.1349

# Generic plot function
def make_plot(x, y, baseline, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', label="Upsample Experiment")
    plt.axhline(y=baseline, color='red', linestyle='--', label="Baseline")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("dataset_usage", filename))
    plt.close()

# === Dataset Usage vs Eval Loss ===
make_plot(usage_vals, eval1_vals, baseline_eval1, "Dataset Usage vs Eval 1 Loss", "Dataset Usage (%)", "usage_eval1.png")
make_plot(usage_vals, eval2_vals, baseline_eval2, "Dataset Usage vs Eval 2 Loss", "Dataset Usage (%)", "usage_eval2.png")
make_plot(usage_vals, eval3_vals, baseline_eval3, "Dataset Usage vs Eval 3 Loss", "Dataset Usage (%)", "usage_eval3.png")

# === Alpha vs Eval Loss ===
make_plot(alpha_vals, eval1_vals, baseline_eval1, "Temperature vs Eval 1 Loss", "Temperature (loss_priority_alpha)", "temp_eval1.png")
make_plot(alpha_vals, eval2_vals, baseline_eval2, "Temperature vs Eval 2 Loss", "Temperature (loss_priority_alpha)", "temp_eval2.png")
make_plot(alpha_vals, eval3_vals, baseline_eval3, "Temperature vs Eval 3 Loss", "Temperature (loss_priority_alpha)", "temp_eval3.png")

# === Beta vs Eval Loss ===
make_plot(beta_vals, eval1_vals, baseline_eval1, "Source Beta vs Eval 1 Loss", "source_balance_beta", "beta_eval1.png")
make_plot(beta_vals, eval2_vals, baseline_eval2, "Source Beta vs Eval 2 Loss", "source_balance_beta", "beta_eval2.png")
make_plot(beta_vals, eval3_vals, baseline_eval3, "Source Beta vs Eval 3 Loss", "source_balance_beta", "beta_eval3.png")
