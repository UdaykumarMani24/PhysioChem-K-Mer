for family in GPCR Hydrolase Ion_Channel Isomerase Kinase Ligase Lyase Structural_Protein Transcription_Factor Transferase; do
    echo "Aligning $family..."
    mafft --auto --quiet ${family}_train.fasta > ${family}_aligned.fasta
    echo "Building profile for $family..."
    hmmbuild hmm_profiles/${family}.hmm ${family}_aligned.fasta
    echo "Done: $family"
    echo "---"
done