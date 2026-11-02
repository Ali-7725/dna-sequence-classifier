# 🧬 BioSeq-AI: DNA Sequence Classifier
> **Introduction to Computational Biology — Assignment 2**  
> An interactive, production-ready machine learning web application for DNA sequence feature extraction, classification, and academic reporting.

---

## 📋 Project Overview & Architecture

BioSeq-AI implements an end-to-end computational biology pipeline featuring:
1. **Bioinformatics Feature Extraction Engine**: Calculates nucleotide counts ($A, T, G, C$), total sequence length ($L$), GC content percentage ($GC\%$), and extended genomic metrics (purine/pyrimidine ratio, CpG motifs, GC skew).
2. **Machine Learning Pipeline**: Trains dynamic Scikit-Learn classifiers (**Random Forest**, **Decision Tree**, **Support Vector Machine**, **Logistic Regression**) on labeled training sequences (S1–S4) and augmented genomic datasets.
3. **Interactive Testing Dashboard**: Real-time sequence prediction, probability confidence gauges, and comparative nucleotide distribution radar/bar plots.
4. **Academic Report Generator**: Pre-formatted, publication-ready answers for Assignment 2 including feature tables, model evaluation metrics, biological feature analysis, real-world pathogen/cancer applications, and critical AI/ML genomics limitations.

---

## 🗂 File Structure

```
dna-sequence-classifier/
├── .streamlit/
│   └── config.toml          # Custom dark Vercel-style UI configuration
├── src/
│   ├── __init__.py
│   ├── feature_extractor.py # Sequence validation & biophysical feature extraction
│   ├── ml_engine.py         # Scikit-Learn training, cross-validation & prediction
│   └── academic_content.py  # Formatted assignment answers & genomic theory
├── app.py                   # Streamlit multi-tab reactive dashboard
├── requirements.txt         # Production Python dependencies
└── README.md                # Documentation and academic reference
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation
Navigate to the project directory and install the required dependencies:

```bash
cd C:\Users\aliab\.gemini\antigravity\scratch\dna-sequence-classifier
pip install -r requirements.txt
```

### 3. Running the Web Application
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```
*Or, if invoking via Python module:*
```bash
python -m streamlit run app.py
```

The application will launch automatically at `http://localhost:8501`.

---

## 📊 Assignment 2 Canonical Feature Table

| Sequence ID | Raw Sequence | Length (bp) | Count A | Count T | Count G | Count C | GC Content (%) | Group Label |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **S1** | `ATGCGATC` | 8 | 2 | 2 | 2 | 2 | **50.00%** | Group A |
| **S2** | `ATGCGTAC` | 8 | 2 | 2 | 2 | 2 | **50.00%** | Group A |
| **S3** | `GCGCGTAT` | 8 | 1 | 2 | 3 | 2 | **62.50%** | Group B |
| **S4** | `GCGCGCGC` | 8 | 0 | 0 | 4 | 4 | **100.00%** | Group B |
| **Test** | `ATGCGCGATC` | 10 | 2 | 2 | 3 | 3 | **60.00%** | **Group B** (Predicted) |

### Key Insight for Test Sequence:
- **Test Sequence:** `ATGCGCGATC` has 3 Guanines and 3 Cytosines out of 10 base pairs $\implies 6 / 10 = \mathbf{60.00\% \text{ GC Content}}$.
- Because Group A sequences exhibit $50.00\%$ GC and Group B sequences exhibit $\ge 62.50\%$ GC, the machine learning model classifies `ATGCGCGATC` into **Group B** with high statistical confidence.

---

## 🧪 Scientific & Machine Learning Logic

### 1. Feature Engineering Formulas
- **GC Content (%):**
  $$\text{GC Content (\%)} = \frac{\text{Count}(G) + \text{Count}(C)}{\text{Total Length}} \times 100$$
- **AT / GC Ratio:**
  $$\text{Ratio} = \frac{\text{Count}(G) + \text{Count}(C)}{\text{Count}(A) + \text{Count}(T)}$$
- **Purine vs Pyrimidine Ratio:**
  $$\text{Ratio}_{\text{pur/pyr}} = \frac{\text{Count}(A) + \text{Count}(G)}{\text{Count}(C) + \text{Count}(T)}$$

### 2. Biological Relevance of GC Content
- **Duplex Thermodynamic Stability:** Guanine and Cytosine pair through **three hydrogen bonds** ($G \equiv C$), whereas Adenine and Thymine pair through **two hydrogen bonds** ($A = T$). Thus, higher GC content significantly elevates melting temperature ($T_m$).
- **CpG Islands and Promoters:** Vertebrate genomes are generally depleted of the `5'-CpG-3'` dinucleotide due to spontaneous deamination of 5-methylcytosine into thymine ($5\text{mC} \rightarrow T$). Elevated GC content often demarcates unmethylated CpG islands in active gene promoters.

### 3. Real-World Genomics Application
- **Pathogen Metagenomics & Taxonomic Binning:** Microbial genomes exhibit characteristic GC content signatures (e.g., *Actinobacteria* $>70\%$ GC vs *Firmicutes* $\sim 30\text{--}38\%$ GC). Alignment-free ML classifiers analyze fragmented sequencing reads from blood or respiratory swabs for ultra-fast point-of-care pathogen identification.
- **Liquid Biopsy Early Cancer Detection:** Detection of aberrant promoter hypermethylation in circulating tumor DNA (ctDNA) for genes such as *BRCA1*, *MLH1*, and *SEPT9*.

### 4. Critical Limitations of AI/ML in Genomics
- **The $n \ll p$ Problem:** Genomic datasets often feature massive feature dimensions with small patient cohort sizes, risking overfitting to sequencing batch artifacts.
- **Loss of Positional Syntax:** Bag-of-nucleotides/count-based models treat sequences without order; an inverted sequence with identical base counts yields the same vector despite encoding totally different biological function.
- **Single-Nucleotide Sensitivity:** A single point mutation or indel can cause a fatal frameshift or disrupt a transcription factor binding motif while barely shifting global GC%.
- **Lack of Epigenetic Context:** Primary 1D sequences cannot account for 3D chromatin conformation (Hi-C) or histone modifications (ChIP-seq).

---

## ☁️ Cloud Deployment Options

### Deploying to Streamlit Community Cloud (Free & Instant)
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository, set `app.py` as the entrypoint, and click **Deploy**.

---

*Authored for Assignment 2: Introduction to Computational Biology.*
