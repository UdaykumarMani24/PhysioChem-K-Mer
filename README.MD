# PhysioChem-K-Mer: A Physicochemical Property-Based Framework for Protein Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.XXXX/xxxxxx-blue)](https://doi.org/10.XXXX/xxxxxx)

This repository contains the implementation for the manuscript:

**"A Theoretical Analysis of PhysioChem-K-Mer Features for Protein Classification using Controlled Synthetic Benchmarks"**

*Keerthika Kamaraj, Senthilkumar Rathnasamy, Udayakumar Mani*  
*School of Chemical and Biotechnology, SASTRA Deemed University*

## 📖 Abstract

Protein family classification remains challenging due to limitations in current k-mer approaches that ignore amino acids' biochemical properties. We present **PhysioChem-K-Mer**, a framework that transforms protein sequences into feature spaces based on physicochemical principles rather than sequence identity alone. Our method achieves **81.33% classification accuracy** (119% relative improvement over standard 3-mer methods) with **73.9% fewer features** and **71.2% faster training times**.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PhysioChem-K-Mer.git
cd PhysioChem-K-Mer

# Install dependencies
pip install -r requirements.txt


## 📊 Key Results
<pre>
| Dataset | Method | Accuracy | Improvement | Features | Training Time |
|---------|--------|----------|-------------|----------|---------------|
| Synthetic (1,500 seqs) | Standard 3-mer | 37.00% | - | 1,000 | 43.2s |
| Synthetic (1,500 seqs) | PhysioChem-Hydropathy | **81.33%** | **+119%** | **261** | **12.5s** |
| Real UniProt (11,620 seqs) | Standard 3-mer | 16.95% | - | 1,000 | 198.95s |
| Real UniProt (11,620 seqs) | PhysioChem-Hydropathy | **64.63%** | **+281%** | **261** | **36.65s** |

</pre>
