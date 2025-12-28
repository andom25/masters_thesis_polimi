#!/usr/bin/env python3
"""
================================================================================
ECT DATA PREPROCESSING - STEP 3: SIGNAL NORMALIZATION
================================================================================

Purpose: ECT Data Preprocessing Pipeline

DESCRIPTION:
    This script performs the final preprocessing step for ECT data. It normalizes the signal values to a [0, 1] range,
    making the data suitable for machine learning models and consistent comparison
    across different measurements.

ALGORITHM:
    Min-Max normalization is applied separately to real and imaginary components:
    
    normalized_value = (value - min) / (max - min)
    
    This transforms the data so that:
    - The minimum value becomes 0
    - The maximum value becomes 1
    - All other values are scaled proportionally between 0 and 1
    
    If min == max (constant signal), all values are set to 0.5 to avoid
    division by zero.

RATIONALE:
    Normalization is essential for:
    1. Machine learning algorithms that are sensitive to feature scales
    2. Comparing signals from different build jobs or channels
    3. Ensuring numerical stability in downstream analysis
    4. Removing arbitrary scale differences between real and imaginary components

INPUT:
    Excel file containing recoater-corrected ECT data (output from Step 2) with:
    - geometry_name: Name of the geometry (CB, DB1, DB2, DB3, DB4, XCT)
    - layer_index: Layer number
    - x1: Spatial position
    - real: Real component of the signal (corrected)
    - imag: Imaginary component of the signal (corrected)

OUTPUT:
    Excel file with normalized real and imaginary values in [0, 1] range

================================================================================
"""

import pandas as pd
import numpy as np
import os


def normalize_signal(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    Normalize a signal column in a DataFrame to a [0, 1] range using min-max scaling.
    
    Formula: normalized = (value - min) / (max - min)
    
    Args:
        df: Input DataFrame containing the signal column
        signal_col: Name of the column to normalize
        
    Returns:
        DataFrame with the specified column normalized to [0, 1] range
        
    Note:
        If all values are identical (max == min), the column is set to 0.5
        to avoid division by zero.
    """
    min_val = df[signal_col].min()
    max_val = df[signal_col].max()
    
    if max_val == min_val:
        df[signal_col] = 0.5  # Avoid division by zero, set to midpoint
    else:
        df[signal_col] = (df[signal_col] - min_val) / (max_val - min_val)
    
    return df


def process_channel_normalization(input_path: str, output_path: str, channel_name: str):
    """
    Process normalization for a single channel.
    
    Normalizes both real and imaginary signal components to [0, 1] range.
    
    Args:
        input_path: Path to input Excel file (recoater-corrected data)
        output_path: Path for output Excel file
        channel_name: Name of the channel (for logging)
    """
    print(f"\n{'=' * 60}")
    print(f"🔄 NORMALIZATION {channel_name}")
    print(f"{'=' * 60}")
    
    print(f"📂 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    df = pd.read_excel(input_path)
    print(f"📊 Loading data...")
    print(f"📈 Dataset: {len(df)} points")
    print(f"📊 Available columns: {df.columns.tolist()}")

    print("\n📊 Original statistics:")
    print(f"  REAL: min={df['real'].min():.6f}, max={df['real'].max():.6f}")
    print(f"  IMAG: min={df['imag'].min():.6f}, max={df['imag'].max():.6f}")

    print("🔧 Normalizing REAL signal...")
    df = normalize_signal(df, 'real')
    print("🔧 Normalizing IMAGINARY signal...")
    df = normalize_signal(df, 'imag')

    print("\n📊 Normalized statistics:")
    print(f"  REAL: min={df['real'].min():.6f}, max={df['real'].max():.6f}")
    print(f"  IMAG: min={df['imag'].min():.6f}, max={df['imag'].max():.6f}")
    
    print("💾 Saving normalized dataset...")
    df.to_excel(output_path, index=False)
    print(f"✅ {channel_name} completed!")
    print(f"📁 File saved: {output_path}")


def main():
    """
    Main function for signal normalization.
    Processes both channels with the corrected files.
    """
    print("🎯 SIGNAL NORMALIZATION (AFTER RECOATER CORRECTION)")
    print("=" * 60)
    
    # Paths for Channel 0 (BJ4)
    ch0_input = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ4/geometrie_excel_BJ4/0_Bj4_peaks.recoater_corrected.xlsx'
    ch0_output = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ4/geometrie_excel_BJ4/0_Bj4_peaks.normalized.xlsx'
    
    # Paths for Channel 1 (BJ4)
    ch1_input = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ4/geometrie_excel_BJ4/1_Bj4_peaks.recoater_corrected.xlsx'
    ch1_output = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ4/geometrie_excel_BJ4/1_Bj4_peaks.normalized.xlsx'
    
    # Process Channel 0
    process_channel_normalization(ch0_input, ch0_output, "CHANNEL 0")
    
    # Process Channel 1
    process_channel_normalization(ch1_input, ch1_output, "CHANNEL 1")
    
    print(f"\n{'=' * 60}")
    print("🎉 ALL CHANNELS HAVE BEEN NORMALIZED SUCCESSFULLY!")
    print("=" * 60)
    print("📊 The output files now contain:")
    print("   • Filtered regions of interest only")
    print("   • Real and imaginary values corrected for recoater effect")
    print("   • Real and imaginary values normalized to [0, 1] range")
    print("   • Separate normalization for real and imaginary components")
    print("   • CB remains unchanged in recoater correction (reference point)")
    print("   • The further from CB, the larger the correction applied")


if __name__ == "__main__":
    main()
