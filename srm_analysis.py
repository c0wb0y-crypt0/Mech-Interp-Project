#!/usr/bin/env python3
"""
SRM Analysis Script - Delta Vector Method
Tests polarity using directional deviation from neutral anchor
This should reveal true antipodal structure in semiotic space

Usage: python srm_analysis.py
Requires: pip install pandas numpy matplotlib scikit-learn sentence-transformers
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = './Results'
OUTPUT_DIR = './Analysis_Output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

def load_csv_safe(path):
    """Load CSV with error handling for mixed types"""
    try:
        df = pd.read_csv(path)
        for col in ['bearish_words', 'neutral_words', 'bullish_words']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def embed_responses(df):
    """Add embedding columns AND polarity delta vectors"""
    print(f"  Embedding {len(df)} responses...")
    
    # Convert to string and clean
    df['bearish_clean'] = df['bearish'].astype(str).str.strip()
    df['neutral_clean'] = df['neutral'].astype(str).str.strip()
    df['bullish_clean'] = df['bullish'].astype(str).str.strip()
    
    # Embed all responses
    bear_embeds = embedder.encode(df['bearish_clean'].tolist(), show_progress_bar=False)
    neut_embeds = embedder.encode(df['neutral_clean'].tolist(), show_progress_bar=False)
    bull_embeds = embedder.encode(df['bullish_clean'].tolist(), show_progress_bar=False)
    
    # Store raw embeddings
    df['bear_embed'] = list(bear_embeds)
    df['neut_embed'] = list(neut_embeds)
    df['bull_embed'] = list(bull_embeds)
    
    # CRITICAL: Compute polarity as DELTA from neutral anchor
    # This extracts the directional component (framing) while canceling semantic content
    df['bear_delta'] = [bear - neut for bear, neut in zip(bear_embeds, neut_embeds)]
    df['bull_delta'] = [bull - neut for bull, neut in zip(bull_embeds, neut_embeds)]
    
    print(f"  → Computed delta vectors (polarity = response - neutral)")
    
    return df

def test_antipodality_delta(df, label=""):
    """Test 1: Antipodal structure using DELTA VECTORS"""
    print(f"\n=== TEST 1: Antipodality via Delta Vectors ({label}) ===")
    print("  Method: Polarity = (response - neutral), cancels semantic content")
    
    # Stack all delta vectors
    bear_deltas = np.stack(df['bear_delta'].values)
    bull_deltas = np.stack(df['bull_delta'].values)
    
    # Mean polarity direction for each frame
    bear_mean_delta = np.mean(bear_deltas, axis=0)
    bull_mean_delta = np.mean(bull_deltas, axis=0)
    
    # Compute cosine between polarity directions
    cos_delta = cosine_similarity([bear_mean_delta], [bull_mean_delta])[0][0]
    
    print(f"  Bearish Δ ↔ Bullish Δ cosine: {cos_delta:.4f}")
    print(f"    → Prediction: Should be NEGATIVE (antipodal polarity)")
    print(f"    → Result: {'✓✓ STRONG ANTIPODALITY' if cos_delta < -0.5 else '✓ CONFIRMED' if cos_delta < -0.2 else '~ WEAK' if cos_delta < 0.2 else '✗ FAILED (positive)'}")
    
    # Additional insight: magnitude of polarity vectors
    bear_mag = np.linalg.norm(bear_mean_delta)
    bull_mag = np.linalg.norm(bull_mean_delta)
    print(f"  Bearish polarity magnitude: {bear_mag:.4f}")
    print(f"  Bullish polarity magnitude: {bull_mag:.4f}")
    print(f"    → Similar magnitudes = symmetric framing strength")
    
    # Compute angle in degrees
    angle = np.arccos(np.clip(cos_delta, -1, 1)) * 180 / np.pi
    print(f"  Angle between polarity vectors: {angle:.1f}°")
    print(f"    → Perfect antipodes = 180°, Orthogonal = 90°, Aligned = 0°")
    
    return {
        'cos_delta': cos_delta,
        'angle': angle,
        'bear_magnitude': bear_mag,
        'bull_magnitude': bull_mag
    }

def test_polarity_strength_distribution(df, label=""):
    """New test: Distribution of polarity strengths"""
    print(f"\n=== BONUS TEST: Polarity Strength Distribution ({label}) ===")
    
    # Compute magnitude of each delta vector
    bear_mags = [np.linalg.norm(delta) for delta in df['bear_delta'].values]
    bull_mags = [np.linalg.norm(delta) for delta in df['bull_delta'].values]
    
    print(f"  Bearish: mean={np.mean(bear_mags):.4f}, std={np.std(bear_mags):.4f}")
    print(f"  Bullish: mean={np.mean(bull_mags):.4f}, std={np.std(bull_mags):.4f}")
    
    # Plot distributions
    plt.figure(figsize=(10, 5))
    plt.hist(bear_mags, bins=20, alpha=0.6, label='Bearish', color='red')
    plt.hist(bull_mags, bins=20, alpha=0.6, label='Bullish', color='green')
    plt.xlabel('Polarity Strength (||Δ||)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Polarity Strength Distribution ({label})', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{label}_polarity_distribution.png', dpi=150)
    print(f"  → Plot saved: {OUTPUT_DIR}/{label}_polarity_distribution.png")
    
    return {'bear_mags': bear_mags, 'bull_mags': bull_mags}

def test_temperature_variance_delta(files_by_temp, label=""):
    """Test 2: Temperature scaling using delta vector variance"""
    print(f"\n=== TEST 2: Temperature-Variance Scaling (Delta Method) ({label}) ===")
    
    temps = sorted(files_by_temp.keys())
    bear_variances = []
    bull_variances = []
    combined_variances = []
    
    for temp in temps:
        df = files_by_temp[temp]
        
        # Variance in polarity direction (delta magnitude)
        bear_mags = [np.linalg.norm(delta) for delta in df['bear_delta'].values]
        bull_mags = [np.linalg.norm(delta) for delta in df['bull_delta'].values]
        
        bear_var = np.var(bear_mags)
        bull_var = np.var(bull_mags)
        combined_var = np.var(bear_mags + bull_mags)
        
        bear_variances.append(bear_var)
        bull_variances.append(bull_var)
        combined_variances.append(combined_var)
        
        print(f"  Temp {temp:.1f}: Combined Var={combined_var:.6f}, Bear={bear_var:.6f}, Bull={bull_var:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(temps, combined_variances, marker='o', linewidth=2.5, markersize=10, label='Combined', color='purple')
    plt.plot(temps, bear_variances, marker='s', linewidth=2, markersize=8, label='Bearish', color='red', alpha=0.7)
    plt.plot(temps, bull_variances, marker='^', linewidth=2, markersize=8, label='Bullish', color='green', alpha=0.7)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Variance in Polarity Strength', fontsize=12)
    plt.title(f'Temperature-Variance Scaling (Delta Method) ({label})', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{label}_temp_variance_delta.png', dpi=150)
    print(f"  → Plot saved: {OUTPUT_DIR}/{label}_temp_variance_delta.png")
    
    # Check correlation
    if len(temps) > 2:
        correlation = np.corrcoef(temps, combined_variances)[0, 1]
        print(f"  Temp-Variance correlation: {correlation:.4f}")
        print(f"    → Prediction: Positive (variance ∝ τ)")
        print(f"    → Result: {'✓ CONFIRMED' if correlation > 0.5 else '~ WEAK' if correlation > 0 else '✗ NEGATIVE'}")
    
    return {'temps': temps, 'variances': combined_variances}

def test_cross_model_delta(df_grok, df_claude, label=""):
    """Test 3: Cross-model convergence using delta vectors"""
    print(f"\n=== TEST 3: Cross-Model Convergence (Delta Method) ({label}) ===")
    
    # Mean polarity deltas for each model
    grok_bear_delta = np.mean(np.stack(df_grok['bear_delta'].values), axis=0)
    grok_bull_delta = np.mean(np.stack(df_grok['bull_delta'].values), axis=0)
    claude_bear_delta = np.mean(np.stack(df_claude['bear_delta'].values), axis=0)
    claude_bull_delta = np.mean(np.stack(df_claude['bull_delta'].values), axis=0)
    
    # Cross-model polarity alignment
    cos_bear = cosine_similarity([grok_bear_delta], [claude_bear_delta])[0][0]
    cos_bull = cosine_similarity([grok_bull_delta], [claude_bull_delta])[0][0]
    
    print(f"  Grok bearish Δ ↔ Claude bearish Δ: {cos_bear:.4f}")
    print(f"  Grok bullish Δ ↔ Claude bullish Δ: {cos_bull:.4f}")
    print(f"    → Prediction: >0.7 (universal polarity structure)")
    print(f"    → Result: {'✓✓ EXTREMELY STRONG' if min(cos_bear, cos_bull) > 0.85 else '✓ STRONG' if min(cos_bear, cos_bull) > 0.7 else '~ MODERATE' if min(cos_bear, cos_bull) > 0.5 else '✗ WEAK'}")
    
    # Additional: Check if Grok bearish ↔ Claude bullish are antipodal (should be negative)
    cross_antipodal = cosine_similarity([grok_bear_delta], [claude_bull_delta])[0][0]
    print(f"  Grok bearish Δ ↔ Claude bullish Δ: {cross_antipodal:.4f}")
    print(f"    → Cross-model antipodality check (should be negative)")
    
    return {
        'cos_bear': cos_bear,
        'cos_bull': cos_bull,
        'cross_antipodal': cross_antipodal
    }

def test_word_polarity_correlation_delta(df, label=""):
    """Test 4: Word count vs polarity strength (using delta magnitude)"""
    print(f"\n=== TEST 4: Word Count vs Polarity Strength (Delta Method) ({label}) ===")
    
    word_counts = []
    polarity_strengths = []
    
    for _, row in df.iterrows():
        # Bearish
        if pd.notna(row['bearish_words']):
            bear_strength = np.linalg.norm(row['bear_delta'])
            word_counts.append(row['bearish_words'])
            polarity_strengths.append(bear_strength)
        
        # Bullish
        if pd.notna(row['bullish_words']):
            bull_strength = np.linalg.norm(row['bull_delta'])
            word_counts.append(row['bullish_words'])
            polarity_strengths.append(bull_strength)
    
    if len(word_counts) > 0:
        correlation = np.corrcoef(word_counts, polarity_strengths)[0, 1]
        
        print(f"  Correlation: {correlation:.4f}")
        print(f"    → Original Hypothesis: Negative (verbose = boundary)")
        print(f"    → Alternative Hypothesis: Positive (verbose = confident framing)")
        print(f"    → Result: {'Supports Alternative (confident elaboration)' if correlation > 0.1 else 'Neutral/Weak signal'}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(word_counts, polarity_strengths, alpha=0.5, s=40, c=polarity_strengths, cmap='viridis')
        plt.colorbar(label='Polarity Strength')
        plt.xlabel('Word Count', fontsize=12)
        plt.ylabel('Polarity Strength (||Δ||)', fontsize=12)
        plt.title(f'Word Count vs Polarity Strength ({label})', fontsize=14)
        
        # Trendline
        z = np.polyfit(word_counts, polarity_strengths, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(word_counts), max(word_counts), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5, label=f'Trend (r={correlation:.3f})')
        
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{label}_word_polarity_delta.png', dpi=150)
        print(f"  → Plot saved: {OUTPUT_DIR}/{label}_word_polarity_delta.png")
        
        return {'correlation': correlation}
    else:
        print("  ✗ No valid word count data")
        return {'correlation': None}

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

print("="*70)
print("SRM ANALYSIS - Delta Vector Method")
print("Extracts polarity as directional deviation from neutral anchor")
print("="*70)

# Load all CSV files
print("\nScanning for result files...")
grok_files = sorted(glob(os.path.join(RESULTS_DIR, 'deception_results_*.csv')))
claude_files = sorted(glob(os.path.join(RESULTS_DIR, 'deception_opus_results_*.csv')))

print(f"  Found {len(grok_files)} Grok files")
print(f"  Found {len(claude_files)} Claude files")

# Load and embed Grok data
print("\n--- Processing Grok Data ---")
grok_by_temp = {}
for f in grok_files:
    temp = float(f.split('temp')[1].split('_')[0])
    df = load_csv_safe(f)
    if df is not None:
        df = embed_responses(df)
        grok_by_temp[temp] = df

# Load and embed Claude data
print("\n--- Processing Claude Data ---")
claude_by_temp = {}
for f in claude_files:
    temp = float(f.split('temp')[1].split('_')[0])
    df = load_csv_safe(f)
    if df is not None:
        df = embed_responses(df)
        claude_by_temp[temp] = df

# Run tests
results = {}

# Test 1: Antipodality with delta vectors (should show negative cosine now!)
if 0.0 in grok_by_temp:
    results['grok_antipodality'] = test_antipodality_delta(grok_by_temp[0.0], "Grok")
if 0.0 in claude_by_temp:
    results['claude_antipodality'] = test_antipodality_delta(claude_by_temp[0.0], "Claude")

# Bonus: Polarity strength distributions
if 0.0 in grok_by_temp:
    test_polarity_strength_distribution(grok_by_temp[0.0], "Grok")
if 0.0 in claude_by_temp:
    test_polarity_strength_distribution(claude_by_temp[0.0], "Claude")

# Test 2: Temperature variance with delta method
if len(grok_by_temp) > 1:
    results['grok_temp_var'] = test_temperature_variance_delta(grok_by_temp, "Grok")
if len(claude_by_temp) > 1:
    results['claude_temp_var'] = test_temperature_variance_delta(claude_by_temp, "Claude")

# Test 3: Cross-model with delta vectors
if 0.0 in grok_by_temp and 0.0 in claude_by_temp:
    results['cross_model'] = test_cross_model_delta(grok_by_temp[0.0], claude_by_temp[0.0], "Grok vs Claude")

# Test 4: Word count correlation with delta magnitude
all_grok = pd.concat(grok_by_temp.values(), ignore_index=True)
all_claude = pd.concat(claude_by_temp.values(), ignore_index=True)
results['grok_word_corr'] = test_word_polarity_correlation_delta(all_grok, "Grok")
results['claude_word_corr'] = test_word_polarity_correlation_delta(all_claude, "Claude")

# Summary
print("\n" + "="*70)
print("SUMMARY - DELTA VECTOR METHOD")
print("="*70)
print("\nKey Insight: Polarity = (Response - Neutral)")
print("  → Cancels semantic similarity, isolates framing direction")
print("\nExpected Improvements:")
print("  1. Antipodality should now show NEGATIVE cosine (< -0.2)")
print("  2. Cross-model alignment should remain strong (validates universality)")
print("  3. Temperature scaling should be clearer")
print("  4. Word count pattern clarifies: verbose = confident vs boundary")
print("\nCheck Analysis_Output/ for updated plots!")
print("="*70)