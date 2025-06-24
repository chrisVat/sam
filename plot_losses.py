import matplotlib.pyplot as plt
import re

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    eval_pattern = re.compile(r"Eval (\d+):.*?Avg Loss = ([\d\.]+)")
    blocks = text.split("\n\n")

    all_evals = []
    for block in blocks:
        matches = eval_pattern.findall(block)
        if matches and "BASE LINE" not in block:
            losses = [float(val) for _, val in sorted(matches, key=lambda x: int(x[0]))]
            if len(losses) == 3:
                all_evals.append(losses)
        elif "BASE LINE" in block:
            baseline = [float(val) for _, val in sorted(matches, key=lambda x: int(x[0]))]

    return all_evals, baseline

def plot_evals(all_evals, baseline):
    x = [1, 2, 3]  # Eval 1, 2, 3

    for idx, eval_set in enumerate(all_evals):
        plt.plot(x, eval_set, label=f'', color='green', alpha=0.5)

    # Baseline
    plt.plot(x, baseline, color='red', linewidth=2, label='Baseline', linestyle='-')

    plt.xticks(x, [f"{i}" for i in x])
    plt.xlabel("Evaluation #")
    plt.ylabel("Average Loss")
    plt.title("Average Loss per Eval (Methods vs Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_report.png")
    plt.show()

# Example usage
log_file = "loss_reports.txt"  # Replace with your filename
evals, baseline = parse_log_file(log_file)
plot_evals(evals, baseline)
