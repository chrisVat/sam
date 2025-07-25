from tqdm import tqdm
import re
import ast
import os
import pickle

REQUIRE_LABELS = False


def parse_batched_entries_split(
    filename,
    eval_only: bool = False,
    train_only: bool = False,
    eval_marker: str = "Beginning Evaluation",
    eval_end_marker: str = "{'eval_loss':"
):
    train_results = []
    eval_results  = []
    in_eval = False
    block = ""

    def _process_block(block_text: str):
        # shuffled IDs
        try:
            a = block_text.index("shuffled id:  tensor([") + len("shuffled id:  tensor([")
            b = block_text.index("], device='cuda:0') og_id: ", a)
            ids_str = block_text[a:b]
            shuffled_ids = [int(x.strip()) for x in ids_str.split(",")]
        except ValueError:
            return

        # OG IDs
        try:
            c = block_text.index("og_id:  tensor([", b) + len("og_id:  tensor([")
            d = block_text.index("], device='cuda:0') loss:  ", c)
            og_str = block_text[c:d]
            og_ids = [int(x.strip()) for x in og_str.split(",")]
        except ValueError:
            return

        # labels matrix
        if REQUIRE_LABELS:
            try:
                e = block_text.index("labels:  tensor([", d) + len("labels:  tensor([")
                f = block_text.index("]], device='cuda:0')", e)
                lbls_str = block_text[e:f]
            except ValueError:
                return

            # split into rows and parse
            labels = []
            for row in lbls_str.split("],"):
                items = row.strip(" []\n")
                if not items:
                    continue
                labels.append([int(x.strip()) for x in items.split(",")])

            if not (len(shuffled_ids) == len(og_ids) == len(labels)):
                return
        else:
            labels = [None] * len(shuffled_ids)

        if not (len(shuffled_ids) == len(og_ids)):
            return

        target = eval_results if in_eval else train_results
        for sid, oid, lbl in zip(shuffled_ids, og_ids, labels):
            target.append((sid, oid, lbl))



    with open(filename, "r") as f:
        for raw in tqdm(f, desc=f"Parsing {filename}"):
            # end‐of‐block indicators

            if eval_end_marker in raw:
                print("Found end of evaluation section.")
                break

            if raw.startswith("global step: ") or raw.startswith("{'loss': ") or raw.startswith("{'eval_loss': "):
                if block:
                    _process_block(block)
                    block = ""
                continue

            # detect eval start
            if eval_marker in raw:
                in_eval = True
                print("Detected evaluation section start.")
                if train_only:
                    break
                continue

            # skip until eval for eval_only
            if eval_only and not in_eval:
                continue

            # new batch?
            if raw.startswith("gpu:  0 shuffled id:"):
                if block:
                    _process_block(block)
                block = raw
            else:
                if block:
                    block += raw

        # final block
        if block:
            _process_block(block)

    return train_results, eval_results



if __name__ == "__main__":
    filename = "best-small-proxy-full.txt"
    filename_s2l = "best-s2l-rel-upsample-6.txt"
    os.makedirs("tmp", exist_ok=True)
    RUN_ANYWAY = True

    if not os.path.exists(f"tmp/{filename}__full_train_results.pkl") or RUN_ANYWAY:
        full_train_results, _ = parse_batched_entries_split(filename, train_only=True)
        with open(f"tmp/{filename}__full_train_results.pkl", "wb") as f:
            pickle.dump(full_train_results, f)
    if not os.path.exists(f"tmp/{filename}__full_test_results.pkl") or RUN_ANYWAY:
        _, full_test_results = parse_batched_entries_split(filename, eval_only=True)
        with open(f"tmp/{filename}__full_test_results.pkl", "wb") as f:
            pickle.dump(full_test_results, f)

    if not os.path.exists(f"tmp/{filename_s2l}__s2l_train_results.pkl") or RUN_ANYWAY:
        s2l_train_results, _ = parse_batched_entries_split(filename_s2l, train_only=True)
        with open(f"tmp/{filename_s2l}__s2l_train_results.pkl", "wb") as f:
            pickle.dump(s2l_train_results, f)
    if not os.path.exists(f"tmp/{filename_s2l}__s2l_test_results.pkl") or RUN_ANYWAY:
        _, s2l_test_results = parse_batched_entries_split(filename_s2l, eval_only=True)
        with open(f"tmp/{filename_s2l}__s2l_test_results.pkl", "wb") as f:
            pickle.dump(s2l_test_results, f)

    with open(f"tmp/{filename}__full_train_results.pkl", "rb") as f:
        full_train_results = pickle.load(f)
    with open(f"tmp/{filename}__full_test_results.pkl", "rb") as f:
        full_test_results = pickle.load(f)
    with open(f"tmp/{filename_s2l}__s2l_train_results.pkl", "rb") as f:
        s2l_train_results = pickle.load(f)
    with open(f"tmp/{filename_s2l}__s2l_test_results.pkl", "rb") as f:
        s2l_test_results = pickle.load(f)

    print(f"Loaded {len(full_train_results)} training and {len(full_test_results)} evaluation results from full model.")
    print(f"Loaded {len(s2l_train_results)} training and {len(s2l_test_results)} evaluation results from S2L model.")
    
    import numpy as np


    def compare_labels(label1, label2):
        if label1 is None and label2 is None:
            return True
        if label1 is None or label2 is None:
            return False
        longer_label = label1 if len(label1) > len(label2) else label2
        shorter_label = label1 if len(label1) <= len(label2) else label2
        return shorter_label == longer_label[:len(shorter_label)]

    # now, convert to dictionary with sid1 as key, then oid1, lbl1 and count (how many times the sid1 appears)
    # each iteration should also confirm that the oid and lbl match for the same sid
    
    test_oids = set(oid for _, oid, _ in full_test_results)

    
    full_dict = {}
    for sid, oid, lbl in full_train_results:
        if oid not in full_dict:
            full_dict[oid] = {'label': lbl, 'count': 0}
        else:
            assert compare_labels(full_dict[oid]['label'], lbl), f"Label mismatch for oid {oid}: {full_dict[oid]['label']} != {lbl}"
        full_dict[oid]['count'] += 1

    s2l_dict = {}
    for sid, oid, lbl in s2l_train_results:
        if oid not in s2l_dict:
            s2l_dict[oid] = {'label': lbl, 'count': 0}
        else:
            assert compare_labels(s2l_dict[oid]['label'], lbl), f"Label mismatch for oid {oid}: {s2l_dict[oid]['label']} != {lbl}"
        s2l_dict[oid]['count'] += 1

    # now compare the dictionaries
    not_found_oids = []
    for oid in full_dict.keys():
        if oid not in s2l_dict:
            not_found_oids.append(oid)
            continue
        assert compare_labels(full_dict[oid]['label'], s2l_dict[oid]['label']), f"Label mismatch for oid {oid}: {full_dict[oid]['label']} != {s2l_dict[oid]['label']}"
        assert oid not in test_oids, f"OID {oid} found in training data and in evaluation data."
    print("oids not found in S2L model:", len(not_found_oids))
    print(not_found_oids[:20], "...")
    

    not_found_oids = []
    for oid in s2l_dict.keys():
        if oid not in full_dict:
            not_found_oids.append(oid)
            continue
        assert compare_labels(s2l_dict[oid]['label'], full_dict[oid]['label']), f"Label mismatch for oid {oid}: {s2l_dict[oid]['label']} != {full_dict[oid]['label']}"
        assert oid not in test_oids, f"OID {oid} found in training data but not in evaluation data."
    print("oids not found in full model:", not_found_oids)
    
    print("All training entries match successfully!")


    """

    full_dict = {}
    for sid, oid, lbl in full_train_results:
        if sid not in full_dict:
            full_dict[sid] = {'oid': oid, 'label': lbl, 'count': 0}
        else:
            assert full_dict[sid]['oid'] == oid, f"Shuffled ID mismatch for sid {sid}: {full_dict[sid]['oid']} != {oid}"
            assert full_dict[sid]['label'] == lbl, f"Label mismatch for sid {sid}: {full_dict[sid]['label']} != {lbl}"
        full_dict[sid]['count'] += 1

    s2l_dict = {}
    for sid, oid, lbl in s2l_train_results:
        if sid not in s2l_dict:
            s2l_dict[sid] = {'oid': oid, 'label': lbl, 'count': 0}
        else:
            assert s2l_dict[sid]['oid'] == oid, f"Shuffled ID mismatch for sid {sid}: {s2l_dict[sid]['oid']} != {oid}"
            assert s2l_dict[sid]['label'] == lbl, f"Label mismatch for sid {sid}: {s2l_dict[sid]['label']} != {lbl}"
        s2l_dict[sid]['count'] += 1

    # now compare the dictionaries
    for sid in full_dict:
        if sid not in s2l_dict:
            print(f"Shuffled ID {sid} found in full model but not in S2L model.")
            continue
        assert full_dict[sid]['oid'] == s2l_dict[sid]['oid'], f"OG ID mismatch for sid {sid}: {full_dict[sid]['oid']} != {s2l_dict[sid]['oid']}"
        assert full_dict[sid]['label'] == s2l_dict[sid]['label'], f"Label mismatch for sid {sid}: {full_dict[sid]['label']} != {s2l_dict[sid]['label']}"
        assert full_dict[sid]['count'] == s2l_dict[sid]['count'], f"Count mismatch for sid {sid}: {full_dict[sid]['count']} != {s2l_dict[sid]['count']}"

    print("All shuffled ID entries match successfully!")
    """

