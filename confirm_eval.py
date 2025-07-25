from tqdm import tqdm
import re

def parse_batched_entries(filename):
    results = []

    # Regex patterns
    id_pattern = re.compile(r"shuffled id:\s+tensor\(\[([^\]]+)\]", re.DOTALL)
    og_pattern = re.compile(r"og_id:\s+tensor\(\[([^\]]+)\]", re.DOTALL)

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("gpu:  0 shuffled id:"):
            # Get entire multi-line entry
            full_block = line
            i += 1
            while i < len(lines) and not lines[i].startswith("gpu:"):
                full_block += lines[i]
                i += 1

            # Parse IDs
            id_match = id_pattern.search(full_block)
            og_match = og_pattern.search(full_block)

            if not (id_match and og_match):
                continue

            shuffled_ids = [int(x.strip()) for x in id_match.group(1).split(",")]
            og_ids = [int(x.strip()) for x in og_match.group(1).split(",")]

            # Parse labels
            labels = []
            label_section = full_block.split("labels:  tensor(")[-1]
            label_lines = label_section.split("],")
            for line in label_lines:
                if "[" in line:
                    nums = re.findall(r"-?\d+", line)
                    if nums:
                        labels.append([int(n) for n in nums])

            # Flatten: zip together individual examples
            for sid, oid, lbl in zip(shuffled_ids, og_ids, labels):
                results.append((sid, oid, lbl))
        else:
            i += 1

    return results

if __name__ == "__main__":
    filename = "DEBUG.txt"
    filename_s2l = "DEBUGs2l.txt"
    full_results = parse_batched_entries(filename)
    s2l_results = parse_batched_entries(filename_s2l)

    """
    for sid, oid, lbl in flat_results:
        print("Shuffled ID:", sid)
        print("OG ID:", oid)
        print("Label[:10]:", lbl[:])
        print("-----")
    """
    # compare the results to ensure they perfectly match
    assert len(full_results) == len(s2l_results), "Results length mismatch!"
    for (sid1, oid1, lbl1), (sid2, oid2, lbl2) in zip(full_results, s2l_results):
        assert sid1 == sid2, f"Shuffled ID mismatch: {sid1} != {sid2}"
        assert oid1 == oid2, f"OG ID mismatch: {oid1} != {oid2}"
        assert lbl1 == lbl2, f"Label mismatch: {lbl1} != {lbl2}"
    print("All entries match successfully!")