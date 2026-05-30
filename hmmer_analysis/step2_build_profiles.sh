#!/bin/bash
# Step 2: Build one HMMER3 profile per family from training sequences
# Usage: bash step2_build_profiles.sh

mkdir -p hmm_profiles
echo "Building HMMER3 profiles..."

while read family; do
    fasta="${family}_train.fasta"
    hmm="hmm_profiles/${family}.hmm"
    
    if [ ! -f "$fasta" ]; then
        echo "  WARNING: $fasta not found, skipping"
        continue
    fi
    
    count=$(grep -c "^>" "$fasta")
    echo "  Building profile for $family ($count sequences)..."
    
    # If only 1 sequence, hmmbuild needs at least 1 — use --singlemx flag
    if [ "$count" -eq 1 ]; then
        hmmbuild --singlemx "$hmm" "$fasta" > /dev/null 2>&1
    else
        hmmbuild "$hmm" "$fasta" > /dev/null 2>&1
    fi
    
    echo "    ? $hmm created"
done < family_list.txt

# Concatenate all profiles into one database
cat hmm_profiles/*.hmm > all_families.hmm
echo ""
echo "All profiles merged into: all_families.hmm"
echo "Done."