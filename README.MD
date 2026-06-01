# PhysioChem-K-Mer: A Physicochemical Property-Based Framework for Protein Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for the manuscript:

**"A Theoretical Analysis of PhysioChem-K-Mer Features for Protein Classification using Controlled Synthetic Benchmarks"**

*Keerthika Kamaraj, Senthilkumar Rathnasamy, Udayakumar Mani*  
*School of Chemical and Biotechnology, SASTRA Deemed University*

---

## 📖 Abstract

Protein family classification remains challenging due to limitations in current k-mer approaches that treat amino acids as categorical tokens without directly encoding physicochemical properties. We present **PhysioChem-K-Mer**, a framework that transforms protein sequences into feature spaces based on physicochemical principles rather than sequence identity alone.

**Key Results:**
- **Synthetic benchmark:** 81.33% accuracy → **+44.33 percentage points** over standard 3-mer (37.00%)
- **Real UniProt data:** 64.63% accuracy → **+47.68 percentage points** over standard 3-mer (16.95%)
- **73.9% fewer features** and **71-81% faster training**
---

## 📊 Key Results


- **Synthetic benchmark:** 81.33% accuracy → **+44.33 percentage points** over standard 3-mer (37.00%)
- **Real UniProt data:** 64.63% accuracy → **+47.68 percentage points** over standard 3-mer (16.95%)
- **73.9% fewer features** and **71-81% faster training**
---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/UdaykumarMani24/PhysioChem-K-Mer.git
cd PhysioChem-K-Mer

# Install dependencies
pip install -r requirements.txt
