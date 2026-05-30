"""
Fixed Step 4 Evaluate — handles header matching robustly
"""
import re
from collections import defaultdict

# ---- 1. Parse HMMER3 tblout: best hit per target sequence ----
hits = defaultdict(list)

with open("hmmer_results.tbl") as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        target = parts[0]   # e.g. test_GPCR|P32745
        query  = parts[2]   # e.g. GPCR_aligned
        evalue = float(parts[4])
        score  = float(parts[5])
        # Clean predicted family: strip _aligned suffix
        pred_family = re.sub(r'_aligned$', '', query)
        hits[target].append((evalue, score, pred_family))

# Best hit per target (lowest E-value)
predictions = {}
for seq_id, hit_list in hits.items():
    best = min(hit_list, key=lambda x: x[0])
    predictions[seq_id] = best[2]

print(f"DEBUG: Total sequences with =1 HMM hit: {len(predictions)}")
if predictions:
    sample = list(predictions.items())[:3]
    print(f"DEBUG: Sample predictions (seq_id -> predicted_family): {sample}")

# ---- 2. Load true labels from test_sequences.fasta ----
true_labels = {}
with open("test_sequences.fasta") as f:
    for line in f:
        if line.startswith(">"):
            header = line.strip().lstrip(">")
            # Header like: test_Kinase|Q9UK32  OR  Kinase|Q9UK32
            raw_family = header.split("|")[0]
            # Strip test_ prefix if present
            family = re.sub(r'^test_', '', raw_family)
            true_labels[header] = family

print(f"DEBUG: Total test sequences loaded: {len(true_labels)}")
if true_labels:
    sample = list(true_labels.items())[:3]
    print(f"DEBUG: Sample true labels (header -> family): {sample}")

# ---- 3. Overlap check ----
pred_keys = set(predictions.keys())
true_keys = set(true_labels.keys())
overlap = pred_keys & true_keys
print(f"DEBUG: Matching keys (pred n true): {len(overlap)}")

# If no overlap, try matching without test_ prefix in keys
if len(overlap) == 0:
    print("DEBUG: No key overlap found. Trying to remap prediction keys...")
    # Try: maybe pred keys have test_ but true_labels keys don't, or vice versa
    # Remap predictions using cleaned keys
    predictions_remap = {}
    for k, v in predictions.items():
        k_clean = re.sub(r'^test_', '', k.split("|")[0]) + "|" + "|".join(k.split("|")[1:])
        predictions_remap[k] = v
        predictions_remap[k_clean] = v
    predictions = predictions_remap
    overlap = set(predictions.keys()) & true_keys
    print(f"DEBUG: After remap, matching keys: {len(overlap)}")

# ---- 4. Calculate metrics ----
families = sorted(set(true_labels.values()))
total = len(true_labels)
correct = 0
no_hit = 0

tp = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)

for seq_id, true_fam in true_labels.items():
    if seq_id in predictions:
        pred_fam = predictions[seq_id]
        if pred_fam == true_fam:
            correct += 1
            tp[true_fam] += 1
        else:
            fp[pred_fam] += 1
            fn[true_fam] += 1
    else:
        no_hit += 1
        fn[true_fam] += 1

accuracy = correct / total * 100

# Per-family F1
f1_scores = []
print(f"\n{'Family':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 80)
for fam in families:
    t = tp[fam]
    p = fp[fam]
    n = fn[fam]
    precision = t / (t + p) if (t + p) > 0 else 0
    recall    = t / (t + n) if (t + n) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)
    print(f"{fam:<30} {t:>6} {p:>6} {n:>6} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}")

macro_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

print("-" * 80)
print(f"\nTotal test sequences : {total}")
print(f"Correct predictions  : {correct}")
print(f"No HMM hit (missed)  : {no_hit}")
print(f"Accuracy             : {accuracy:.2f}%")
print(f"Macro F1-Score       : {macro_f1:.2f}%")
print(f"\nHMMER3 Accuracy  : {accuracy:.2f}%")
print(f"HMMER3 Macro F1  : {macro_f1:.2f}%")

with open("hmmer3_results_summary.txt", "w") as f:
    f.write(f"Total sequences: {total}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"No hit: {no_hit}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"Macro F1: {macro_f1:.2f}%\n")

print("\nSummary saved to: hmmer3_results_summary.txt")