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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from math import pi
from itertools import product
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']   # Blue, Orange, Green, Purple
METHOD_LABELS = ['Standard\n3-mer', 'PhysioChem\nBiochemical',
                 'PhysioChem\nHydropathy', 'PhysioChem\nStructural']

def save(name):
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✔ {name} saved")
    
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

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')

def load_dataset(path):
    df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
    # Normalise column names
    col_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_lower)
    for col in df.columns:
        if 'seq' in col and 'id' not in col and 'len' not in col:
            df = df.rename(columns={col: 'Sequence'})
        elif 'fam' in col:
            df = df.rename(columns={col: 'Family'})
    df['Sequence'] = df['Sequence'].str.upper().str.strip()
    df['Family']   = df['Family'].str.strip().str.lower()
    return df[['Sequence', 'Family']].dropna()


def clean_dataset(df):
    n0 = len(df)
    df = df[df['Sequence'].apply(lambda s: set(s).issubset(VALID_AAS))].copy()
    df = df[df['Sequence'].apply(len) >= 50].copy()
    df = df.drop_duplicates(subset='Sequence').reset_index(drop=True)
    print(f"  Cleaned: {n0} → {len(df)} sequences")
    print(f"  Families ({df['Family'].nunique()}): {sorted(df['Family'].unique())}")
    return df

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
    df = pd.read_excel('uniprot_10k_dataset.xlsx')
    sequences = df['Sequence'].tolist()
    labels = df['Family'].tolist()

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

    return pd.DataFrame(results), classifiers, X_test, y_test
# ─────────────────────────────────────────────────────────────────────────────
# 6.  ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(df, scheme="scheme_enhanced"):

    seqs = df["Sequence"].tolist()
    labels = df["Family"].tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        seqs,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    class AblClf(EnhancedPhysioChemKmerClassifier):

        def __init__(self, sch, comp, trans, ent):

            super().__init__(scheme=sch)

            self._c = comp
            self._t = trans
            self._e = ent

        def _calculate_sequence_features(self, sequences):

            all_features = []

            properties = sorted(
                list(set(self.mapping.values()))
            )

            all_possible_kmers = [
                "".join(combo)
                for combo in product(properties, repeat=self.k)
            ]

            for seq in sequences:

                prop_seq = self._sequence_to_property_string(seq)

                features = []

                # Always include k-mer frequencies
                kmers = [
                    prop_seq[i:i+self.k]
                    for i in range(len(prop_seq)-self.k+1)
                ]

                kmer_counter = Counter(kmers)

                total_kmers = len(kmers)

                if total_kmers > 0:
                    kmer_freqs = [
                        kmer_counter.get(k, 0)/total_kmers
                        for k in all_possible_kmers
                    ]
                else:
                    kmer_freqs = [0]*len(all_possible_kmers)

                features.extend(kmer_freqs)

                # Composition
                if self._c:

                    aa_counts = Counter(seq)

                    for prop in properties:

                        prop_count = sum(
                            aa_counts.get(aa, 0)
                            for aa in self.mapping
                            if self.mapping[aa] == prop
                        )

                        features.append(
                            prop_count / len(seq)
                        )

                # Transitions
                if self._t:

                    transitions = [
                        prop_seq[i] + prop_seq[i+1]
                        for i in range(len(prop_seq)-1)
                    ]

                    trans_counter = Counter(transitions)

                    for p1 in properties:
                        for p2 in properties:

                            t = p1 + p2

                            features.append(
                                trans_counter.get(t, 0)/len(transitions)
                                if len(transitions) > 0 else 0
                            )

                # Entropy
                if self._e:

                    prop_counts = Counter(prop_seq)

                    entropy = 0

                    for count in prop_counts.values():

                        p = count / len(prop_seq)

                        entropy -= p * np.log2(p)

                    runs = []

                    current_run = 1

                    for i in range(1, len(prop_seq)):

                        if prop_seq[i] == prop_seq[i-1]:
                            current_run += 1
                        else:
                            runs.append(current_run)
                            current_run = 1

                    runs.append(current_run)

                    features.append(entropy)
                    features.append(np.mean(runs))
                    features.append(max(runs))

                all_features.append(features)

            # Correct feature names
            self.feature_names = []

            self.feature_names.extend(
                [f"kmer_{k}" for k in all_possible_kmers]
            )

            if self._c:
                self.feature_names.extend(
                    [f"comp_{p}" for p in properties]
                )

            if self._t:
                self.feature_names.extend(
                    [f"trans_{p1}{p2}" for p1 in properties for p2 in properties]
                )

            if self._e:
                self.feature_names.extend(
                    ["entropy", "avg_run", "max_run"]
                )

            return np.array(all_features)

        def get_num_features(self):
            return len(self.feature_names)

    configs = [
        ("k-mer only", False, False, False),
        ("k-mer + Composition", True, False, False),
        ("k-mer + Transitions", False, True, False),
        ("k-mer + Entropy", False, False, True),
        ("Full (all features)", True, True, True),
    ]

    print(f"\nAblation ({scheme}):")

    rows = []

    for label, c, t, e in configs:

        clf = AblClf(
            scheme,
            c,
            t,
            e
        )

        clf.fit(X_tr, y_tr)

        yp = clf.predict(X_te)

        acc = accuracy_score(y_te, yp)

        f1 = f1_score(
            y_te,
            yp,
            average="weighted"
        )

        nf = clf.get_num_features()

        print(
            f"{label:<30} "
            f"Acc={acc:.4f} "
            f"F1={f1:.4f} "
            f"Features={nf}"
        )

        rows.append({
            "Config": label,
            "Accuracy": acc,
            "F1_Score": f1,
            "Num_Features": nf
        })

    return pd.DataFrame(rows)


from sklearn.metrics import confusion_matrix, classification_report
from math import pi


def create_publication_figures(results_df, classifiers, X_test, y_test, df, ablation_df):
    
    plt.style.use("ggplot")

    method_names = results_df["Method"].tolist()

    # ==========================================================
    # FIGURE 1 — EDA
    # ==========================================================

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    df["Family"].value_counts().plot(
        kind="bar",
        ax=axes[0]
    )
    axes[0].set_title("A) Class Distribution")
    axes[0].set_ylabel("Number of Sequences")
    axes[0].set_xlabel("Protein Family")
    axes[0].tick_params(axis="x", rotation=90)
    
    seq_lengths = df["Sequence"].apply(len)

    axes[1].hist(
    seq_lengths,
    bins=50,
    edgecolor="black",
    alpha=0.85
    )

    axes[1].set_xlim(
    1,
    np.percentile(seq_lengths, 99)
    )

    axes[1].set_title("B) Sequence Length Distribution")
    axes[1].set_xlabel("Sequence Length (aa, log scale)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    family_lengths = df.groupby("Family")["Sequence"].apply(
        lambda x: np.mean([len(i) for i in x])
    )

    family_lengths.plot(
        kind="bar",
        ax=axes[2]
    )

    axes[2].set_title("C) Avg Sequence Length")
    axes[2].set_ylabel("Average Length")
    axes[2].set_xlabel("Protein Family")
    axes[2].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.savefig("fig1_eda.png", dpi=300)
    plt.close()

    # ==========================================================
    # FIGURE 2 — Schema dimensions
    # ==========================================================

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
    results_df["Method"],
    results_df["Num_Features"],
    color=["gray", "orange", "green", "red"]
        )
    for bar in bars:
        plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height()+10,
        int(bar.get_height()),
        ha="center"
        )
        
    plt.title("Feature Space Reduction Across Encoding Schemes")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.ylabel("Number of Features")

    plt.tight_layout()
    plt.savefig("fig2_schema_dims.png", dpi=300)
    plt.close()

    # ==========================================================
    # FIGURE 3 — Performance
    # ==========================================================

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics = [
    ("Accuracy", "Accuracy", (0,0)),
    ("F1_Score", "F1 Score", (0,1)),
    ("Training_Time", "Training Time (s)", (1,0)),
    ("Num_Features", "Feature Count", (1,1))
    ]

    for metric, title, pos in metrics:

        ax = axes[pos]

        bars = ax.bar(
            results_df["Method"],
            results_df[metric],
            alpha = 0.85
        )

        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=30)

        for bar, val in zip(bars, results_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + (max(results_df[metric]) * 0.02),
                round(val, 3),
                ha="center"
            )

    plt.tight_layout()
    plt.savefig("fig3_performance.png", dpi=300)
    plt.close()

    # ==========================================================
    # FIGURE 4 — All confusion matrices
    # ==========================================================

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for ax, method in zip(axes.flatten(), method_names):

        clf = classifiers[method]

        y_pred = clf.predict(X_test)

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=sorted(set(y_test))
        )

        sns.heatmap(
            cm,
            xticklabels=sorted(set(y_test)),
            yticklabels=sorted(set(y_test)),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        acc = results_df.loc[
        results_df["Method"] == method,
        "Accuracy"
        ].values[0]

        ax.set_title(f"{method}\nAcc={acc:.3f}")

    plt.tight_layout()
    plt.savefig("fig4_all_confusion.png", dpi=300)
    plt.close()

    # ==========================================================
    # FIGURE 5 — Per-family F1
    # ==========================================================

    heatmap_rows = []

    for method in method_names:

        clf = classifiers[method]

        y_pred = clf.predict(X_test)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division = 0
        )

        row = {}

        for family in sorted(set(y_test)):
            row[family] = report[family]["f1-score"]

        heatmap_rows.append(row)

    per_family_df = pd.DataFrame(
    heatmap_rows,
    index=method_names
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        per_family_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5
    )

    plt.title("Per-family F1 Score")
    plt.xlabel("Protein Family")
    plt.ylabel("Method")

    plt.tight_layout()
    plt.savefig("fig5_per_family_f1.png", dpi=300)
    plt.close()
    
    # ==========================================================
    # FIGURE 6 — Ablation study
    # ==========================================================

    plt.figure(figsize=(10,6))

    bars = plt.bar(
    ablation_df["Config"],
    ablation_df["Accuracy"],
    color="teal"
    )

    for bar in bars:
        plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height(),
        f"{bar.get_height():.3f}",
        ha="center"
        )
    
    plt.ylim(0, 1)

    plt.xticks(rotation=30)
    plt.ylabel("Accuracy")
    plt.title("Ablation Study")

    plt.tight_layout()
    plt.savefig(
    "fig6_ablation.png",
    dpi=300
    )

    plt.close()


    # ==========================================================
    # FIGURE 7 — Feature importance
    # ==========================================================

    physio_methods = [
        m for m in method_names
        if m != "Standard_3mer_RF"
    ]

    fig, axes = plt.subplots(
        1,
        len(physio_methods),
        figsize=(18, 7)
    )

    for ax, method in zip(axes, physio_methods):

        top_feats = classifiers[method].get_feature_importance(10)

        names = [x[0] for x in top_feats]
        vals = [x[1] for x in top_feats]

        ax.barh(
        names[::-1],
        vals[::-1]
        )
        
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f"{v:.3f}")

        ax.set_title(method)

    plt.tight_layout()
    plt.savefig("fig7_feature_importance.png", dpi=300)
    plt.close()

    # ==========================================================
    # FIGURE 8 — Radar chart
    # ==========================================================

    categories = [
        "Accuracy",
        "F1_Score",
        "Accuracy per Feature"
    ]

    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in results_df.iterrows():

        efficiency_raw = row["Accuracy"] / row["Num_Features"]

        efficiency = (
            efficiency_raw /
            (results_df["Accuracy"] / results_df["Num_Features"]).max()
        )

        values = [
            row["Accuracy"],
            row["F1_Score"],
            efficiency
        ]

        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=row["Method"]
        )
        
        ax.fill(
        angles,
        values,
        alpha=0.1
        )

    plt.xticks(
        angles[:-1],
        categories
    )

    plt.legend(loc="upper right")

    plt.savefig("fig8_radar.png", dpi=300)
    plt.close()

    # ==========================================================
    # FIGURE 9 — Summary table
    # ==========================================================

    fig, ax = plt.subplots(figsize=(16, 3))

    ax.axis("off")

    table = ax.table(
        cellText=results_df.round(4).values,
        colLabels=results_df.columns,
        loc="center"
    )
    
    table.auto_set_column_width(
        col=list(range(len(results_df.columns)))
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)

    plt.savefig("fig9_summary_table.png", dpi=300)
    plt.close()
    
    
# Run the enhanced analysis
print("RUNNING ENHANCED PHYSIOCHEM-K-MER ANALYSIS...")
print("="*70)

enhanced_results, enhanced_classifiers, X_test, y_test = run_enhanced_analysis()


results_df = pd.DataFrame(enhanced_results)

df = pd.read_excel("uniprot_10k_dataset.xlsx")

physio_only = enhanced_results[
    enhanced_results["Method"] != "Standard_3mer_RF"
]

best_method = physio_only.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]["Method"]

method_to_scheme = {
    "PhysioChem_Enhanced": "scheme_enhanced",
    "PhysioChem_Hydropathy": "scheme_hydropathy",
    "PhysioChem_Structural": "scheme_structural"
}

best_scheme = method_to_scheme[best_method]

ablation_df = run_ablation(
    df,
    scheme=best_scheme
)

create_publication_figures(    
    enhanced_results,
    enhanced_classifiers,
    X_test,
    y_test,
    df,
    ablation_df
)

print("\n" + "="*80)
print("ENHANCED RESULTS SUMMARY")
print("="*80)
print(enhanced_results.round(4).to_string(index=False))

# Create enhanced visualization
df = pd.read_excel("uniprot_10k_dataset.xlsx")

print(ablation_df)

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

# Metrics
accuracy_ratio = (
    best_physiochem["Accuracy"] /
    best_standard["Accuracy"]
)

relative_improvement = (
    best_physiochem["Accuracy"] -
    best_standard["Accuracy"]
) / best_standard["Accuracy"]

feature_ratio = (
    best_physiochem["Num_Features"] /
    best_standard["Num_Features"]
)

time_ratio = (
    best_physiochem["Training_Time"] /
    best_standard["Training_Time"]
)


print("\nPerformance Analysis:")

print(f"  Absolute Accuracy (Standard): {best_standard['Accuracy']:.3f}")
print(f"  Absolute Accuracy (PhysioChem): {best_physiochem['Accuracy']:.3f}")

print(
    f"\n  Relative Accuracy Improvement: "
    f"+{relative_improvement:.1%} over standard baseline"
)

print(
    f"  Accuracy Gain: "
    f"{accuracy_ratio:.2f}x over standard baseline"
)

print(
    f"  Feature Reduction: "
    f"{(1 - feature_ratio):.1%} fewer features"
)

print(
    f"  Training Time Reduction: "
    f"{(1 - time_ratio):.1%} faster training"
)

if accuracy_ratio > 0.8:
    print("SUCCESS: PhysioChem achieves competitive performance with interpretable features!")
elif accuracy_ratio > 0.6:
    print("\nMODERATE: PhysioChem shows promise but needs further optimization.")
else:
    print("\nNEEDS IMPROVEMENT: PhysioChem performance needs significant enhancement.")
