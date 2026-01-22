#!/usr/bin/env python3
"""
SRM Paper-Ready Manifold Visualizer - Full Temperature Support
Generates figures for ALL temperatures, saves per-temp, and combines into series
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Config
RESULTS_DIR = './Results'
OUTPUT_DIR = './Paper_Figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Temperatures to process
TEMPS = ['0.0', '0.3', '0.7', '1.0']

print("="*70)
print("SRM PAPER-READY FIGURE GENERATOR - FULL TEMP SUPPORT")
print("="*70 + "\n")

# Load embedder
print("Loading sentence transformer for polarity analysis...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Ready\n")

# ============================================================================
# LOAD DATA PER TEMPERATURE
# ============================================================================

def load_data_for_temp(temp):
    """Load all probe results for a specific temperature"""
    print(f"Loading data for temp {temp}...")
    files = glob(os.path.join(RESULTS_DIR, f'*temp{temp}*.csv'))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in ['bearish', 'neutral', 'bullish']):
                dfs.append(df)
                print(f"  ✓ Loaded {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"  ⚠ Skip {os.path.basename(f)}: {e}")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  → Total rows for temp {temp}: {len(combined)}\n")
        return combined
    else:
        print(f"  → No data for temp {temp}\n")
        return pd.DataFrame()

data_by_temp = {temp: load_data_for_temp(temp) for temp in TEMPS}

# ============================================================================
# PER-TEMP FIGURES
# ============================================================================

per_temp_figures = []

for temp, df in data_by_temp.items():
    if df.empty:
        continue
    
    print(f"\nGenerating figures for temp {temp}...")
    
    # Embed for polarity strength (distance from neutral)
    bear_emb = embedder.encode(df['bearish'].tolist(), show_progress_bar=False)
    neut_emb = embedder.encode(df['neutral'].tolist(), show_progress_bar=False)
    bull_emb = embedder.encode(df['bullish'].tolist(), show_progress_bar=False)
    
    # Polarity strength: avg cosine distance from neutral
    bear_strength = cosine_similarity(bear_emb, neut_emb).diagonal()
    bull_strength = cosine_similarity(bull_emb, neut_emb).diagonal()
    combined_strength = np.mean([bear_strength, bull_strength], axis=0)
    
    # Figure: Polarity Strength Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(bear_strength, kde=True, color='red', alpha=0.6, label='Bearish', ax=ax)
    sns.histplot(bull_strength, kde=True, color='green', alpha=0.6, label='Bullish', ax=ax)
    ax.set_title(f'Polarity Strength Distribution (Temp {temp})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Cosine Distance from Neutral')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    per_temp_path = f'{OUTPUT_DIR}/Polarity_Distribution_Temp{temp}.png'
    plt.savefig(per_temp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-temp: {per_temp_path}")
    per_temp_figures.append(per_temp_path)
    
    # Additional per-temp figure: Variance scaling proxy (strength spread)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(len(combined_strength)), combined_strength, alpha=0.7, color='purple')
    ax.set_title(f'Combined Polarity Strength per Response (Temp {temp})')
    ax.set_xlabel('Response Index')
    ax.set_ylabel('Polarity Strength')
    variance_path = f'{OUTPUT_DIR}/Polarity_Variance_Temp{temp}.png'
    plt.savefig(variance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved variance: {variance_path}")

# ============================================================================
# COMBINED SERIES FIGURE (All Temps in One)
# ============================================================================

print("\nGenerating combined temperature series figure...")
if per_temp_figures:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, temp in enumerate(TEMPS):
        if i >= len(per_temp_figures):
            break
        img = plt.imread(per_temp_figures[i])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Temperature {temp}', fontsize=14, fontweight='bold')
    
    plt.suptitle('Polarity Strength Distributions Across Temperatures', fontsize=18, fontweight='bold', y=0.98)
    combined_path = f'{OUTPUT_DIR}/Combined_Temperature_Series.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined series: {combined_path}")
else:
    print("No per-temp figures to combine")

print("\n" + "="*70)
print("ALL DONE! Check Paper_Figures/ for per-temp + combined visualizations")
print("="*70)