
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel('uniprot_10k_dataset.xlsx')

sequences = df['Sequence'].tolist()
labels = df['Family'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

with open('training_sequences.fasta', 'w') as f:
    for seq, family in zip(X_train, y_train):
        f.write(f">{family}\n")
        f.write(f"{seq}\n")

with open('test_sequences.fasta', 'w') as f:
    for seq, family in zip(X_test, y_test):
        f.write(f">test_{family}\n")
        f.write(f"{seq}\n")

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

