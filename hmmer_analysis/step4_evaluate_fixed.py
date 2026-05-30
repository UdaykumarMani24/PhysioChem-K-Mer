import re
from collections import defaultdict

hits = defaultdict(list)

with open("hmmer_results.tbl") as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        target = parts[0]
        query  = parts[2]
        evalue = float(parts[4])
        score  = float(parts[5])
        pred_family = re.sub(r'_aligned$', '', query)
        hits[target].append((evalue, score, pred_family))

predictions = {}
for seq_id, hit_list in hits.items():
    best = min(hit_list, key=lambda x: x[0])
    predictions[seq_id] = best[2]

print(f"DEBUG hits found: {len(predictions)}")
sample = list(predictions.items())[:3]
print(f"DEBUG sample predictions: {sample}")

true_labels = {}
with open("test_sequences.fasta") as f:
    for line in f:
        if line.startswith(">"):
            header = line.strip().lstrip(">")
            raw_family = header.split("|")[0]
            family = re.sub(r'^test_', '', raw_family)
            true_labels[header] = family

print(f"DEBUG test sequences loaded: {len(true_labels)}")
sample2 = list(true_labels.items())[:3]
print(f"DEBUG sample true labels: {sample2}")

overlap = set(predictions.keys()) & set(true_labels.keys())
print(f"DEBUG matching keys: {len(overlap)}")

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
f1_scores = []
print(f"\n{'Family':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 80)
for fam in families:
    t = tp[fam]; p = fp[fam]; n = fn[fam]
    precision = t / (t + p) if (t + p) > 0 else 0
    recall    = t / (t + n) if (t + n) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)
    print(f"{fam:<30} {t:>6} {p:>6} {n:>6} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}")

macro_f1 = sum(f1_scores) / len(f1_scores) * 100
print("-" * 80)
print(f"\nTotal: {total}  Correct: {correct}  No hit: {no_hit}")
print(f"Accuracy     : {accuracy:.2f}%")
print(f"Macro F1     : {macro_f1:.2f}%")

with open("hmmer3_results_summary.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}%\nMacro F1: {macro_f1:.2f}%\n")
print("Done.")
