
import pandas as pd

# Read BLAST results
df = pd.read_csv('blast_results.txt', sep='\t', header=None,
                 names=['query', 'subject', 'pident', 'evalue', 'stitle'])

# Keep only first hit per query
df = df.drop_duplicates(subset='query', keep='first')

print(f'Total test sequences: {len(df)}')

# Extract true family from query (format: test_Family|Protein_ID)
def extract_true_family(q):
    # Remove 'test_' prefix
    without_test = q.replace('test_', '')
    # Get family before '|'
    family = without_test.split('|')[0]
    return family

# Extract predicted family from subject (format: Family|Protein_ID)
def extract_pred_family(s):
    family = s.split('|')[0]
    return family

df['true_family'] = df['query'].apply(extract_true_family)
df['pred_family'] = df['subject'].apply(extract_pred_family)

# Calculate accuracy
df['correct'] = df['true_family'] == df['pred_family']
accuracy = df['correct'].mean() * 100

print('='*60)
print('BLAST CLASSIFICATION RESULTS')
print('='*60)
print(f'Total test sequences: {len(df)}')
print(f'Correctly classified: {df["correct"].sum()}')
print(f'Accuracy: {accuracy:.2f}%')
print('='*60)

# Show sample predictions
print('\nSample predictions (first 10):')
for i in range(min(10, len(df))):
    row = df.iloc[i]
    print(f'  {row["query"]} -> {row["pred_family"]} (true: {row["true_family"]})')
