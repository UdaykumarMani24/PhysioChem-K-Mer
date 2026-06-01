# PhysioChem-K-Mer: A Physicochemical Property-Based Framework for Protein Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for the manuscript:

**"A Theoretical Analysis of PhysioChem-K-Mer Features for Protein Classification using Controlled Synthetic Benchmarks"**

*Keerthika Kamaraj, Senthilkumar Rathnasamy, Udayakumar Mani*  
*School of Chemical and Biotechnology, SASTRA Deemed University*

---

## 📖 Abstract

Standard k-mer methods treat amino acids as categorical tokens without directly encoding physicochemical properties. Although physicochemical properties have been incorporated into various bioinformatics tasks, their potential as a direct, systematic alternative for the k-mer counting paradigm has not been fully evaluated. We present PhysioChem-K-mer, a framework that transforms protein sequences into physicochemical property-based feature spaces, serving as an alternative to conventional amino-acid-identity k-mer representations. Our main hypothesis is that property-based representations capture functional constraints more effectively than traditional amino acid-based methods. To test this hypothesis, we created a controlled benchmark comprising 1,500 synthetic sequences spanning 10 diverse protein families. The dataset retained core functional motifs while deliberately excluding evolutionary patterns typically found in natural biological sequences. Notably, our hydropathy-based PhysioChem-K-mer achieved a classification accuracy of 81.33% on a controlled synthetic benchmark, representing an absolute gain of 44.33 percentage points over standard 3-mer methods (37.00%). The framework was further evaluated using real UniProt/Swiss-Prot data, comprising 11,620 sequences across 10 families, to ensure practical generalizability. Based on real data, PhysioChem-Hydropathy achieved 64.63%, an absolute gain of 47.68 percentage points over the standard 3-mer baseline (16.95%), while reducing features by 73.9% and training time by 81.6%. By directly integrating biochemical knowledge into feature representations as a primary design principle, PhysioChem-K-mer combines interpretability with computational efficiency. These results suggest that physicochemical properties offer a vital source of information for protein classification, validated here on both synthetic and real-world data.

**Key Results:**
- **Synthetic benchmark:** 81.33% accuracy → **+44.33 percentage points** over standard 3-mer (37.00%)
- **Real UniProt data:** 64.63% accuracy → **+47.68 percentage points** over standard 3-mer (16.95%)
- **73.9% fewer features** and **71-81% faster training**

-
---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/UdaykumarMani24/PhysioChem-K-Mer.git
cd PhysioChem-K-Mer

# Install dependencies
pip install -r requirements.txt
