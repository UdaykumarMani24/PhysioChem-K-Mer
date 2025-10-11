import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Enhanced physicochemical grouping schemes
PHYSICOCHEMICAL_GROUPS = {
    'scheme_enhanced': {
        'A': 'GAVLIMFWP',  # Aliphatic/Hydrophobic
        'C': 'C',          # Cysteine (special)
        'S': 'STY',        # Polar/Serine-like
        'N': 'NQ',         # Amide
        'D': 'DE',         # Acidic
        'R': 'KRH',        # Basic
        'P': 'P',          # Proline (special)
    },
    'scheme_hydropathy': {
        'H': 'AVILMFW',    # Hydrophobic
        'N': 'GTP',        # Neutral
        'P': 'STYNQ',      # Polar
        'C': 'DE',         # Charged negative
        'B': 'KRH',        # Charged positive
        'S': 'C',          # Special (Cysteine)
    },
    'scheme_structural': {
        'S': 'GAVLIMFWP',  # Structural/buried
        'P': 'STYNQ',      # Polar/surface
        'C': 'DEKRH',      # Charged
        'X': 'C',          # Cross-linking
    }
}

class EnhancedPhysioChemKmerClassifier:
    def __init__(self, scheme='scheme_enhanced', k=3, model_type='rf', use_composition=True, use_entropy=True):
        self.scheme = scheme
        self.k = k
        self.model_type = model_type
        self.use_composition = use_composition
        self.use_entropy = use_entropy
        self.mapping = self._create_mapping_dict(PHYSICOCHEMICAL_GROUPS[scheme])
        self.feature_names = None

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=2000, random_state=42, C=0.1, class_weight='balanced')
        else:
            self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, class_weight='balanced')

    def _create_mapping_dict(self, groups):
        mapping = {}
        for code, aas in groups.items():
            for aa in aas:
                mapping[aa] = code
        return mapping

    def _sequence_to_property_string(self, sequence):
        return ''.join(self.mapping.get(aa, 'X') for aa in sequence)

    def _calculate_sequence_features(self, sequences):
 
        all_features = []

        # Generate all possible k-mers for this scheme
        properties = list(set(self.mapping.values()))
        from itertools import product
        all_possible_kmers = [''.join(combo) for combo in product(properties, repeat=self.k)]

        for seq in sequences:
            prop_seq = self._sequence_to_property_string(seq)
            features = []

            # 1. Basic k-mer frequencies
            if len(prop_seq) >= self.k:
                kmers = [prop_seq[i:i+self.k] for i in range(len(prop_seq)-self.k+1)]
                kmer_counts = {kmer: kmers.count(kmer) for kmer in all_possible_kmers}
                total_kmers = len(kmers)
                kmer_freqs = [kmer_counts[kmer]/total_kmers if total_kmers > 0 else 0 for kmer in all_possible_kmers]
                features.extend(kmer_freqs)
            else:
                features.extend([0] * len(all_possible_kmers))

            # 2. Amino acid composition features
            if self.use_composition:
                aa_counts = Counter(seq)
                total_aas = len(seq)
                for prop in properties:
                    prop_count = sum(aa_counts.get(aa, 0) for aa in self.mapping.keys() if self.mapping[aa] == prop)
                    features.append(prop_count / total_aas if total_aas > 0 else 0)

            # 3. Transition probabilities between property groups
            if len(prop_seq) > 1:
                transitions = []
                for i in range(len(prop_seq)-1):
                    transition = prop_seq[i] + prop_seq[i+1]
                    transitions.append(transition)

                unique_transitions = set(transitions)
                for prop1 in properties:
                    for prop2 in properties:
                        transition = prop1 + prop2
                        features.append(transitions.count(transition) / len(transitions) if len(transitions) > 0 else 0)

            # 4. Sequence entropy and complexity
            if self.use_entropy and len(prop_seq) > 0:
                # Shannon entropy of property distribution
                prop_counts = Counter(prop_seq)
                entropy = 0
                for count in prop_counts.values():
                    p = count / len(prop_seq)
                    if p > 0:
                        entropy -= p * np.log2(p)
                features.append(entropy)

                # Property group runs (consecutive same properties)
                runs = []
                current_run = 1
                for i in range(1, len(prop_seq)):
                    if prop_seq[i] == prop_seq[i-1]:
                        current_run += 1
                    else:
                        runs.append(current_run)
                        current_run = 1
                runs.append(current_run)
                features.append(np.mean(runs) if runs else 0)
                features.append(max(runs) if runs else 0)

            all_features.append(features)

        # Set feature names
        self.feature_names = []
        self.feature_names.extend([f"kmer_{kmer}" for kmer in all_possible_kmers])
        if self.use_composition:
            self.feature_names.extend([f"comp_{prop}" for prop in properties])
        if len(properties) > 0:
            self.feature_names.extend([f"trans_{p1}{p2}" for p1 in properties for p2 in properties])
        if self.use_entropy:
            self.feature_names.extend(["entropy", "avg_run", "max_run"])

        return np.array(all_features)

    def fit(self, sequences, labels):
        X = self._calculate_sequence_features(sequences)
        self.model.fit(X, labels)
        return self

    def predict(self, sequences):
        X = self._calculate_sequence_features(sequences)
        return self.model.predict(X)

    def get_feature_importance(self, top_n=10):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return []

        return sorted(zip(self.feature_names, importances), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    def get_num_features(self):
        return len(self.feature_names) if self.feature_names else 0

def run_enhanced_analysis():


    # Load dataset
    df = pd.read_csv('protein_family_dataset.csv')
    sequences = df['sequence'].tolist()
    labels = df['family'].tolist()

    print(f"Dataset loaded: {len(sequences)} sequences, {len(set(labels))} families")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")

    # Define enhanced methods to compare
    methods = {
        'Standard_3mer_RF': {'scheme': None, 'model': 'rf', 'enhanced': False},
        'PhysioChem_Enhanced': {'scheme': 'scheme_enhanced', 'model': 'rf', 'enhanced': True},
        'PhysioChem_Hydropathy': {'scheme': 'scheme_hydropathy', 'model': 'rf', 'enhanced': True},
        'PhysioChem_Structural': {'scheme': 'scheme_structural', 'model': 'rf', 'enhanced': True},
    }

    results = []
    classifiers = {}

    for method_name, params in methods.items():
        print(f"\n{'='*60}")
        print(f"Training {method_name}")
        print(f"{'='*60}")

        start_time = time.time()

        if params['scheme'] is None:
            # Standard k-mer with Random Forest
            class StandardKmerClassifier:
                def __init__(self, max_features=1000):
                    self.max_features = max_features
                    self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
                    self.feature_names = None

                def _extract_standard_kmers(self, sequences, k=3):
                    all_kmers = set()
                    for seq in sequences:
                        if len(seq) >= k:
                            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                            all_kmers.update(kmers)

                    if len(all_kmers) > self.max_features:
                        kmer_counts = Counter()
                        for seq in sequences:
                            if len(seq) >= k:
                                kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                                kmer_counts.update(kmers)
                        self.feature_names = [kmer for kmer, count in kmer_counts.most_common(self.max_features)]
                    else:
                        self.feature_names = sorted(all_kmers)

                    feature_vectors = []
                    for seq in sequences:
                        if len(seq) >= k:
                            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                        else:
                            kmers = []
                        kmer_count = {kmer: kmers.count(kmer) for kmer in self.feature_names}
                        total = sum(kmer_count.values())
                        if total > 0:
                            feature_vector = [kmer_count[kmer]/total for kmer in self.feature_names]
                        else:
                            feature_vector = [0] * len(self.feature_names)
                        feature_vectors.append(feature_vector)

                    return np.array(feature_vectors)

                def fit(self, sequences, labels):
                    X = self._extract_standard_kmers(sequences)
                    self.model.fit(X, labels)
                    return self

                def predict(self, sequences):
                    X = self._extract_standard_kmers(sequences)
                    return self.model.predict(X)

                def get_num_features(self):
                    return len(self.feature_names) if self.feature_names else 0

            classifier = StandardKmerClassifier(max_features=1000)
        else:
            # Enhanced PhysioChem classifier
            classifier = EnhancedPhysioChemKmerClassifier(
                scheme=params['scheme'],
                k=3,
                model_type=params['model'],
                use_composition=True,
                use_entropy=True
            )

        # Train model
        classifier.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predict
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        prediction_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            'Method': method_name,
            'Accuracy': accuracy,
            'F1_Score': f1,
            'Training_Time': training_time,
            'Prediction_Time': prediction_time,
            'Num_Features': classifier.get_num_features()
        })

        classifiers[method_name] = classifier

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Prediction Time: {prediction_time:.2f}s")
        print(f"Number of Features: {classifier.get_num_features()}")

        # Show top features for PhysioChem methods
        if params['scheme'] is not None and hasattr(classifier, 'get_feature_importance'):
            print(f"\nTop 5 features for {method_name}:")
            top_features = classifier.get_feature_importance(top_n=5)
            for feature, importance in top_features:
                biological_meaning = {
                    'AAA': 'Aliphatic core', 'CCC': 'Cysteine-rich', 'SSS': 'Serine-like cluster',
                    'NNN': 'Amide-rich', 'DDD': 'Acidic region', 'RRR': 'Basic region',
                    'HHH': 'Hydrophobic core', 'PPP': 'Polar surface', 'BBB': 'Basic cluster',
                    'comp_A': 'Aliphatic composition', 'comp_R': 'Basic composition',
                    'comp_D': 'Acidic composition', 'entropy': 'Sequence diversity',
                    'avg_run': 'Property homogeneity'
                }.get(feature, 'Functional pattern')
                print(f"  {feature}: {importance:.4f} ({biological_meaning})")

    return pd.DataFrame(results), classifiers

def create_enhanced_visualization(results_df, classifiers, X_test, y_test):

    plt.figure(figsize=(18, 12))

    # Colors for methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Performance comparison
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(results_df))
    width = 0.35

    bars1 = plt.bar(x_pos - width/2, results_df['Accuracy'], width, label='Accuracy',
                   alpha=0.8, color=colors[:len(results_df)])
    bars2 = plt.bar(x_pos + width/2, results_df['F1_Score'], width, label='F1-Score',
                   alpha=0.8, color=colors[:len(results_df)])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('A) Enhanced Classification Performance', fontweight='bold', fontsize=12)
    plt.xticks(x_pos, results_df['Method'], rotation=45)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    # 2. Computational efficiency (log scale for better visualization)
    plt.subplot(2, 3, 2)
    methods = results_df['Method'].tolist()
    training_times = results_df['Training_Time'].tolist()
    prediction_times = results_df['Prediction_Time'].tolist()

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width/2, training_times, width, label='Training Time',
            color=colors[:len(methods)], alpha=0.7)
    plt.bar(x + width/2, prediction_times, width, label='Prediction Time',
            color=colors[:len(methods)], alpha=0.7)

    plt.xlabel('Method')
    plt.ylabel('Time (seconds)')
    plt.title('B) Computational Efficiency', fontweight='bold', fontsize=12)
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)

    # 3. Feature space comparison
    plt.subplot(2, 3, 3)
    feature_counts = results_df['Num_Features'].tolist()
    bars = plt.bar(methods, feature_counts, color=colors[:len(methods)], alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Method')
    plt.ylabel('Number of Features')
    plt.title('C) Feature Space Dimensionality', fontweight='bold', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 4. Performance vs Features scatter plot
    plt.subplot(2, 3, 4)
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.scatter(row['Num_Features'], row['Accuracy'], s=100, color=colors[i], alpha=0.7, label=row['Method'])
        plt.annotate(row['Method'], (row['Num_Features'], row['Accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('D) Accuracy vs Feature Complexity', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 5. Best method confusion matrix
    plt.subplot(2, 3, 5)
    best_method = results_df.loc[results_df['Accuracy'].idxmax(), 'Method']
    best_classifier = classifiers[best_method]
    y_pred = best_classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_test)),
                yticklabels=sorted(set(y_test)))
    plt.title(f'E) Confusion Matrix - {best_method}', fontweight='bold', fontsize=12)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 6. Key insights summary
    plt.subplot(2, 3, 6)
    plt.axis('off')

    best_row = results_df.loc[results_df['Accuracy'].idxmax()]
    insights = [
        "KEY INSIGHTS:",
        "",
        f"Best Method: {best_row['Method']}",
        f"Best Accuracy: {best_row['Accuracy']:.3f}",
        f"Features Used: {best_row['Num_Features']}",
        f"Training Time: {best_row['Training_Time']:.2f}s",
        "",
        "PhysioChem Advantages:",
        "- Biologically interpretable",
        "- Reduced feature space",
        "- Faster computation",
        "- Domain knowledge integration"
    ]

    plt.text(0.05, 0.95, '\n'.join(insights), transform=plt.gca().transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig('enhanced_physiochem_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the enhanced analysis
print("RUNNING ENHANCED PHYSIOCHEM-K-MER ANALYSIS...")
print("="*70)

enhanced_results, enhanced_classifiers = run_enhanced_analysis()

print("\n" + "="*80)
print("ENHANCED RESULTS SUMMARY")
print("="*80)
print(enhanced_results.round(4).to_string(index=False))

# Create enhanced visualization
create_enhanced_visualization(enhanced_results, enhanced_classifiers,
                             pd.read_csv('protein_family_dataset.csv')['sequence'].tolist()[1200:1500],
                             pd.read_csv('protein_family_dataset.csv')['family'].tolist()[1200:1500])

# Detailed analysis
print("\n" + "="*80)
print("DETAILED PERFORMANCE ANALYSIS")
print("="*80)

best_standard = enhanced_results[enhanced_results['Method'] == 'Standard_3mer_RF'].iloc[0]
best_physiochem = enhanced_results[enhanced_results['Method'] != 'Standard_3mer_RF'].sort_values('Accuracy', ascending=False).iloc[0]

print(f"Standard K-mer Performance:")
print(f"  Accuracy: {best_standard['Accuracy']:.3f}")
print(f"  Features: {best_standard['Num_Features']}")
print(f"  Training Time: {best_standard['Training_Time']:.2f}s")

print(f"\nBest PhysioChem Performance ({best_physiochem['Method']}):")
print(f"  Accuracy: {best_physiochem['Accuracy']:.3f}")
print(f"  Features: {best_physiochem['Num_Features']}")
print(f"  Training Time: {best_physiochem['Training_Time']:.2f}s")

accuracy_ratio = best_physiochem['Accuracy'] / best_standard['Accuracy']
feature_ratio = best_physiochem['Num_Features'] / best_standard['Num_Features']
time_ratio = best_physiochem['Training_Time'] / best_standard['Training_Time']

print(f"\nPerformance Ratios:")
print(f"  Accuracy: {accuracy_ratio:.1%} of standard")
print(f"  Features: {feature_ratio:.1%} of standard")
print(f"  Time: {time_ratio:.1%} of standard")

if accuracy_ratio > 0.8:
    print("SUCCESS: PhysioChem achieves competitive performance with interpretable features!")
elif accuracy_ratio > 0.6:
    print("\nMODERATE: PhysioChem shows promise but needs further optimization.")
else:
    print("\NNEEDS IMPROVEMENT: PhysioChem performance needs significant enhancement.")