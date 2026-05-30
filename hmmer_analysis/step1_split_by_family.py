"""
Step 1: Split training_sequences.fasta into one file per family
Usage: python step1_split_by_family.py training_sequences.fasta
"""
import sys
from collections import defaultdict

input_file = sys.argv[1] if len(sys.argv) > 1 else "training_sequences.fasta"

families = defaultdict(list)
current_header = None
current_seq = []

with open(input_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_header:
                family = current_header.split("|")[0].lstrip(">")
                families[family].append((current_header, "".join(current_seq)))
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)
    # Last sequence
    if current_header:
        family = current_header.split("|")[0].lstrip(">")
        families[family].append((current_header, "".join(current_seq)))

print(f"Found {len(families)} families:")
for fam, seqs in sorted(families.items()):
    fname = f"{fam}_train.fasta"
    with open(fname, "w") as out:
        for header, seq in seqs:
            out.write(f"{header}\n{seq}\n")
    print(f"  {fam}: {len(seqs)} sequences ? {fname}")

# Save family list for later scripts
with open("family_list.txt", "w") as f:
    for fam in sorted(families.keys()):
        f.write(fam + "\n")

print("\nDone. Family list saved to family_list.txt")