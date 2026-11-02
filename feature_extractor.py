"""
DNA Feature Extraction Engine
Computational Biology - Sequence Analysis Module
"""

import re
from typing import Dict, Any, Tuple, List
import pandas as pd


VALID_NUCLEOTIDES = set("ATGC")
IUPAC_AMBIGUOUS = set("RYSWKMBDHVN")


def sanitize_sequence(seq: str) -> str:
    """
    Cleans sequence: uppercase, removes whitespace, newlines, and FASTA headers.
    """
    if not seq:
        return ""
    # Strip FASTA header if present
    lines = seq.strip().splitlines()
    filtered_lines = [line.strip() for line in lines if not line.strip().startswith(">")]
    clean_seq = "".join(filtered_lines).upper()
    # Remove any non-alphabetic whitespace or dashes
    clean_seq = re.sub(r"[^A-Z]", "", clean_seq)
    return clean_seq


def validate_dna_sequence(seq: str) -> Tuple[bool, str, List[str]]:
    """
    Validates if sequence contains only canonical DNA nucleotides (A, T, G, C).
    Returns: (is_valid, message, invalid_chars)
    """
    clean_seq = sanitize_sequence(seq)
    if not clean_seq:
        return False, "Sequence is empty.", []

    invalid_chars = sorted(list(set(clean_seq) - VALID_NUCLEOTIDES))
    if invalid_chars:
        if all(c in IUPAC_AMBIGUOUS for c in invalid_chars):
            return True, f"Sequence contains IUPAC ambiguous bases: {', '.join(invalid_chars)}", invalid_chars
        return False, f"Invalid characters detected: {', '.join(invalid_chars)}. Only A, T, G, C permitted.", invalid_chars

    return True, "Valid canonical DNA sequence.", []


def extract_features(seq: str, seq_id: str = "Input_Seq", include_extended: bool = True) -> Dict[str, Any]:
    """
    Extracts core biological and quantitative features from a DNA sequence:
    Core Requirements:
      - Nucleotide counts (A, T, G, C)
      - Total sequence length
      - GC content percentage (%)

    Extended Biological Features:
      - AT content percentage (%)
      - GC / AT ratio
      - Purine (A+G) vs Pyrimidine (C+T) count & ratio
      - CpG dinucleotide count & Observed/Expected ratio
      - GC skew: (G - C) / (G + C)
    """
    clean_seq = sanitize_sequence(seq)
    length = len(clean_seq)

    if length == 0:
        return {
            "ID": seq_id,
            "Sequence": "",
            "Length": 0,
            "Count_A": 0,
            "Count_T": 0,
            "Count_G": 0,
            "Count_C": 0,
            "GC_Content_%": 0.0,
            "AT_Content_%": 0.0,
            "GC_AT_Ratio": 0.0,
            "Purine_Count": 0,
            "Pyrimidine_Count": 0,
            "Purine_Pyrimidine_Ratio": 0.0,
            "CpG_Count": 0,
            "GC_Skew": 0.0,
        }

    count_a = clean_seq.count("A")
    count_t = clean_seq.count("T")
    count_g = clean_seq.count("G")
    count_c = clean_seq.count("C")

    # Core requirements
    gc_count = count_g + count_c
    at_count = count_a + count_t
    gc_content_pct = round((gc_count / length) * 100.0, 2)
    at_content_pct = round((at_count / length) * 100.0, 2)

    # Extended biological features
    gc_at_ratio = round(gc_count / at_count, 3) if at_count > 0 else float("inf")
    purines = count_a + count_g
    pyrimidines = count_c + count_t
    pur_pyr_ratio = round(purines / pyrimidines, 3) if pyrimidines > 0 else float("inf")
    
    # CpG dinucleotide count
    cpg_count = clean_seq.count("CG")
    
    # GC skew: (G - C) / (G + C)
    gc_skew = round((count_g - count_c) / gc_count, 3) if gc_count > 0 else 0.0

    features = {
        "ID": seq_id,
        "Sequence": clean_seq,
        "Length": length,
        "Count_A": count_a,
        "Count_T": count_t,
        "Count_G": count_g,
        "Count_C": count_c,
        "GC_Content_%": gc_content_pct,
        "AT_Content_%": at_content_pct,
    }

    if include_extended:
        features.update({
            "GC_AT_Ratio": 999.0 if gc_at_ratio == float("inf") else gc_at_ratio,
            "Purines (A+G)": purines,
            "Pyrimidines (C+T)": pyrimidines,
            "Purine/Pyrimidine_Ratio": 999.0 if pur_pyr_ratio == float("inf") else pur_pyr_ratio,
            "CpG_Count": cpg_count,
            "GC_Skew": gc_skew,
        })

    return features


def extract_batch_features(sequences: List[Tuple[str, str, str]], include_extended: bool = False) -> pd.DataFrame:
    """
    Processes a list of tuples: (seq_id, sequence, label) into a structured DataFrame.
    """
    records = []
    for item in sequences:
        if len(item) == 3:
            seq_id, seq, label = item
        else:
            seq_id, seq = item[0], item[1]
            label = "Unknown"
        
        feat = extract_features(seq, seq_id=seq_id, include_extended=include_extended)
        feat["Label"] = label
        records.append(feat)

    df = pd.DataFrame(records)
    # Reorder columns nicely
    core_cols = ["ID", "Sequence", "Label", "Length", "Count_A", "Count_T", "Count_G", "Count_C", "GC_Content_%"]
    other_cols = [c for c in df.columns if c not in core_cols]
    return df[core_cols + other_cols]
