
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel('uniprot_10k_dataset.xlsx')

print(f"Dataset loaded: {len(df)} sequences")
print(f"Columns: {df.columns.tolist()}")

# Get columns
sequences = df['Sequence'].tolist()
labels = df['Family'].tolist()
protein_ids = df['Protein_ID'].tolist()

# Split into training (80%) and test (20%)
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    sequences, labels, protein_ids, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTraining: {len(X_train)} sequences")
print(f"Test: {len(X_test)} sequences")

# Save training sequences with UNIQUE headers (Family|Protein_ID)
with open('training_sequences.fasta', 'w') as f:
    for seq, family, pid in zip(X_train, y_train, ids_train):
        f.write(f">{family}|{pid}\n")
        f.write(f"{seq}\n")

# Save test sequences with UNIQUE headers (test_Family|Protein_ID)
with open('test_sequences.fasta', 'w') as f:
    for seq, family, pid in zip(X_test, y_test, ids_test):
        f.write(f">test_{family}|{pid}\n")
        f.write(f"{seq}\n")

print("\n? FASTA files created with UNIQUE headers!")
print("   - training_sequences.fasta")
print("   - test_sequences.fasta")
