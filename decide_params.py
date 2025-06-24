import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from itertools import product
import numpy as np

# === Load Log File ===
log_path = "loss_reports.txt"
with open(log_path, 'r') as f:
    raw_log = f.read()
print(raw_log)

# === Step 1: Parse Logs ===
entries = []
pattern = re.compile(
    r"(\d+):\s*(.*?)Eval 1: .*?Avg Loss = ([\d.]+)\s*Eval 2: .*?Avg Loss = ([\d.]+)\s*Eval 3: .*?Avg Loss = ([\d.]+)",
    re.DOTALL
)

for match in pattern.finditer(raw_log):
    idx, param_str, loss1, loss2, loss3 = match.groups()
    params = {
        "loss_priority_alpha": None,
        "source_balance_beta": None,
        "max_unique": None,
        "max_dup_per_example": None
    }
    for key in params:
        found = re.search(rf"{key}:\s*([\d.]+)", param_str)
        if found:
            params[key] = float(found.group(1))
    params["avg_loss"] = float(loss3) 
    entries.append(params)

df = pd.DataFrame(entries)

# === Step 2: Fill Baseline Values ===
baseline_values = df.dropna().mean(numeric_only=True)
df_filled = df.fillna(baseline_values)

# === Step 3: Train Non-linear Model ===
X = df_filled.drop(columns="avg_loss")
y = df_filled["avg_loss"]
model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X, y)

# === Step 4: Feature Importance ===
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# === Step 5: Predict Over Grid of Parameters ===
loss_priority_alpha_vals = [0.25, 0.5, 0.75]
source_balance_beta_vals = [0.25, 0.5, 0.75, 1.0]
max_unique_vals = [0.5, 0.75, 0.8]
max_dup_per_example_vals = [10]  # fixed

grid = list(product(loss_priority_alpha_vals, source_balance_beta_vals, max_unique_vals, max_dup_per_example_vals))
grid_df = pd.DataFrame(grid, columns=X.columns)
grid_df["predicted_loss"] = model.predict(grid_df)


# Define fine-grained values
alpha_vals = np.round(np.arange(0.01, 1.01, 0.01), 2)
beta_vals = np.round(np.arange(0.01, 1.01, 0.01), 2)
max_unique_vals = [0.5, 0.75, 0.8]
max_dup_per_example_vals = [10]  # fixed in your logs


# Generate all combinations
fine_grid = list(product(alpha_vals, beta_vals, max_unique_vals, max_dup_per_example_vals))
fine_grid_df = pd.DataFrame(fine_grid, columns=["loss_priority_alpha", "source_balance_beta", "max_unique", "max_dup_per_example"])

# Predict using your trained model
fine_grid_df["predicted_loss"] = model.predict(fine_grid_df)

# Find and print the best config
best_fine_config = fine_grid_df.sort_values("predicted_loss").head(1)
print(best_fine_config)




# === Step 6: Show Top Real Configs ===
best_real_config = df_filled.sort_values("avg_loss").head(5)
print("\nTop Actual Configurations (from real evaluations):")
print(best_real_config)

# === Step 7: Plot Real Effects Only ===
plt.figure(figsize=(6, 4))
plt.scatter(df_filled["source_balance_beta"], df_filled["avg_loss"], c='blue')
plt.xlabel("source_balance_beta")
plt.ylabel("Avg Loss")
plt.title("Actual Loss vs. source_balance_beta")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_loss_vs_source_balance_beta.png")

plt.figure(figsize=(6, 4))
plt.scatter(df_filled["loss_priority_alpha"], df_filled["avg_loss"], c='red')
plt.xlabel("loss_priority_alpha")
plt.ylabel("Avg Loss")
plt.title("Actual Loss vs. loss_priority_alpha")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_loss_vs_loss_priority_alpha.png")

plt.figure(figsize=(6, 4))
plt.scatter(df_filled["max_unique"], df_filled["avg_loss"], c='green')
plt.xlabel("max_unique")
plt.ylabel("Avg Loss")
plt.title("Actual Loss vs. max_unique")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_loss_vs_max_unique.png")
