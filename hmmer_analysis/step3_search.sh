#!/bin/bash
# Step 3: Search all test sequences against all HMM profiles
# Usage: bash step3_search.sh

echo "Searching test sequences against HMM profiles..."
hmmsearch --tblout hmmer_results.tbl \
          --noali \
          -E 1e-3 \
          --cpu 4 \
          all_families.hmm \
          test_sequences.fasta > hmmer_full_output.txt 2>&1

echo "Results saved to: hmmer_results.tbl"
echo "Full output saved to: hmmer_full_output.txt"
echo "Done."