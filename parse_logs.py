import re
import math

def parse_eval_losses(log_path, min_eval_len=10):
    def extract_id_and_loss(line):
        match = re.search(r"id:\s*tensor\(\[(\d+)\]", line)
        loss_match = re.search(r"loss:\s*([\d\.eE+-]+|nan)", line)
        if match and loss_match:
            try:
                id_val = int(match.group(1))
                loss_val = float(loss_match.group(1))
                if math.isnan(loss_val):
                    return id_val, None
                return id_val, loss_val
            except ValueError:
                return None, None
        return None, None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    evals = []
    i = 0
    skip_count = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("[Eval] Skipping batch"):
            skip_count += 1
            i += 1
            continue

        id_val, loss_val = extract_id_and_loss(line)

        if id_val == 0:
            potential_eval = []
            j = i
            expected_id = 0
            while j < len(lines):
                curr_line = lines[j]
                if curr_line.startswith("[Eval] Skipping batch"):
                    skip_count += 1
                    j += 1
                    continue

                next_id, next_loss = extract_id_and_loss(curr_line)
                if next_id == expected_id:
                    if next_loss is not None:
                        potential_eval.append(next_loss)
                    expected_id += 1
                    j += 1
                else:
                    break

            if len(potential_eval) >= min_eval_len:
                evals.append(potential_eval)
                i = j
                continue

        i += 1

    for idx, losses in enumerate(evals):
        avg = sum(losses) / len(losses) if losses else float('nan')
        print(f"Eval {idx + 1}: {len(losses)} entries, Avg Loss = {avg:.4f}")
    
    print(f"Skipped {skip_count} batch lines.")

# Example usage
file_1 = "s2lup_14.txt"
file_2 = "s2luporiginal_13.txt"
parse_eval_losses(file_2)
