#!/usr/bin/env python3
"""
================================================================================
ECT DATA PREPROCESSING - STEP 2: RECOATER EFFECT CORRECTION
================================================================================
Purpose: ECT Data Preprocessing Pipeline

DESCRIPTION:
    This script corrects the recoater effect in ECT data. The recoater mechanism in powder bed fusion additive manufacturing
    creates a systematic spatial trend in the ECT measurements that needs to be
    removed for accurate analysis.

ALGORITHM:
    1. Calculate a global trend using linear regression on stable layers (90-100)
       - These layers are considered "stable" because the powder bed has settled
       - Uses all data points from geometries CB, DB1, DB2, DB3, DB4
    
    2. Compute linear regression coefficients (slope and intercept) for both
       the real and imaginary signal components
    
    3. Apply the correction to ALL layers using the computed global trend:
       - The CB geometry position is used as the reference point
       - The further a measurement point is from CB, the larger the correction
       - Correction formula: corrected_value = original_value - slope * (x_pos - CB_pos)
    
    4. Process both real and imaginary signal components separately

RATIONALE:
    The recoater moves across the build platform in the x-direction, creating
    a linear trend in the ECT measurements. By using stable layers to estimate
    this trend and removing it, we obtain cleaner signals that better represent
    the actual material properties.

INPUT:
    Excel file containing filtered ECT data (output from Step 1) with columns:
    - geometry_name: Name of the geometry (CB, DB1, DB2, DB3, DB4, XCT)
    - layer_index: Layer number
    - x1: Spatial position
    - real: Real component of the signal
    - imag: Imaginary component of the signal

OUTPUT:
    - Corrected Excel file with recoater effect removed
    - Visualization plots showing before/after correction for REAL signal
    - Visualization plots showing before/after correction for IMAGINARY signal

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def analyze_recoater_effect_global(df: pd.DataFrame, 
                                   stable_layers: List[int] = None) -> Dict:
    """
    Analyze the recoater effect using linear regression on stable layers.
    
    Computes the slope through linear regression on all points from stable layers.
    Uses geometries CB, DB1, DB2, DB3, DB4 for robust estimation.
    
    Args:
        df: Input DataFrame containing ECT data
        stable_layers: List of layer indices to use for trend calculation
                      (default: layers 90-100)
    
    Returns:
        Dictionary containing regression parameters and statistics:
        - global_real_slope: Slope of real signal trend
        - global_imag_slope: Slope of imaginary signal trend
        - real_intercept, imag_intercept: Y-intercepts
        - real_r2, imag_r2: R-squared values for regression quality
        - avg_cb_position: Average x-position of CB geometry (reference point)
    """
    print("🔍 Analyzing recoater effect (linear regression on stable layers)...")
    
    if stable_layers is None:
        stable_layers = list(range(90, 101))  # Layers 90-100
    
    print(f"📊 Stable layers used: {stable_layers}")
    
    # Collect all points from stable layers
    all_x_positions = []
    all_real_values = []
    all_imag_values = []
    
    for layer in stable_layers:
        layer_data = df[df['layer_index'] == layer]
        
        # Filter only geometries of interest (CB, DB1, DB2, DB3, DB4)
        geometries = ['CB', 'DB1', 'DB2', 'DB3', 'DB4']
        layer_filtered = layer_data[layer_data['geometry_name'].isin(geometries)]
        
        if len(layer_filtered) == 0:
            print(f"⚠️ Layer {layer}: no valid data, skipped")
            continue
        
        # Add all points from this layer
        all_x_positions.extend(layer_filtered['x1'].tolist())
        all_real_values.extend(layer_filtered['real'].tolist())
        all_imag_values.extend(layer_filtered['imag'].tolist())
        
        print(f"  Layer {layer}: {len(layer_filtered)} points added")
    
    if len(all_x_positions) == 0:
        print("❌ No valid points found!")
        return None
    
    # Convert to numpy arrays for regression
    x_array = np.array(all_x_positions)
    real_array = np.array(all_real_values)
    imag_array = np.array(all_imag_values)
    
    print(f"\n📊 Total points for regression: {len(x_array)}")
    
    # Calculate linear regression for REAL component
    # y = mx + b -> real_slope = m, real_intercept = b
    real_coeffs = np.polyfit(x_array, real_array, 1)  # degree 1 = straight line
    real_slope = real_coeffs[0]
    real_intercept = real_coeffs[1]
    
    # Calculate linear regression for IMAGINARY component
    imag_coeffs = np.polyfit(x_array, imag_array, 1)
    imag_slope = imag_coeffs[0]
    imag_intercept = imag_coeffs[1]
    
    # Calculate R² to evaluate regression quality
    real_predicted = real_slope * x_array + real_intercept
    imag_predicted = imag_slope * x_array + imag_intercept
    
    real_ss_res = np.sum((real_array - real_predicted) ** 2)
    real_ss_tot = np.sum((real_array - np.mean(real_array)) ** 2)
    real_r2 = 1 - (real_ss_res / real_ss_tot) if real_ss_tot != 0 else 0
    
    imag_ss_res = np.sum((imag_array - imag_predicted) ** 2)
    imag_ss_tot = np.sum((imag_array - np.mean(imag_array)) ** 2)
    imag_r2 = 1 - (imag_ss_res / imag_ss_tot) if imag_ss_tot != 0 else 0
    
    # Calculate average CB position for reference
    cb_data = df[df['geometry_name'] == 'CB']
    avg_cb_position = cb_data['x1'].mean()
    
    print(f"\n📈 LINEAR REGRESSION COMPUTED:")
    print(f"   REAL slope: {real_slope:.6f}")
    print(f"   REAL intercept: {real_intercept:.6f}")
    print(f"   REAL R²: {real_r2:.6f}")
    print(f"   IMAG slope: {imag_slope:.6f}")
    print(f"   IMAG intercept: {imag_intercept:.6f}")
    print(f"   IMAG R²: {imag_r2:.6f}")
    print(f"   Average CB position: {avg_cb_position:.3f}")
    print(f"   Layers used: {len(stable_layers)}")
    
    return {
        'global_real_slope': real_slope,
        'global_imag_slope': imag_slope,
        'real_intercept': real_intercept,
        'imag_intercept': imag_intercept,
        'real_r2': real_r2,
        'imag_r2': imag_r2,
        'avg_cb_position': avg_cb_position,
        'stable_layers': stable_layers,
        'total_points': len(x_array),
        'x_positions': x_array,
        'real_values': real_array,
        'imag_values': imag_array
    }


def correct_recoater_effect_global(df: pd.DataFrame, global_trend: Dict) -> pd.DataFrame:
    """
    Correct the recoater effect using the global trend.
    
    The correction is applied as follows:
    - CB remains unchanged (reference point)
    - The further from CB, the larger the correction applied
    - Correction: value -= slope * (x_position - CB_position)
    
    Args:
        df: Input DataFrame containing ECT data
        global_trend: Dictionary with regression parameters from analyze_recoater_effect_global
        
    Returns:
        DataFrame with corrected real and imaginary values
    """
    print("🔧 Correcting recoater effect (global trend)...")
    
    if global_trend is None:
        print("❌ Global trend not available!")
        return df
    
    df_corrected = df.copy()
    
    # Extract global trend parameters
    global_real_slope = global_trend['global_real_slope']
    global_imag_slope = global_trend['global_imag_slope']
    avg_cb_position = global_trend['avg_cb_position']
    
    print(f"📊 Applying global trend:")
    print(f"   REAL slope: {global_real_slope:.6f}")
    print(f"   IMAG slope: {global_imag_slope:.6f}")
    print(f"   Reference CB position: {avg_cb_position:.3f}")
    
    # Apply correction to all points
    for idx in df_corrected.index:
        x_pos = df_corrected.loc[idx, 'x1']
        
        # Calculate correction based on global trend
        # Correction is proportional to distance from average CB position
        real_correction = global_real_slope * (x_pos - avg_cb_position)
        imag_correction = global_imag_slope * (x_pos - avg_cb_position)
        
        # Subtract correction to remove the global slope
        df_corrected.loc[idx, 'real'] -= real_correction
        df_corrected.loc[idx, 'imag'] -= imag_correction
    
    print(f"✅ Global correction applied to {len(df_corrected)} data points")
    
    return df_corrected


def visualize_correction_global(df_original: pd.DataFrame, df_corrected: pd.DataFrame, 
                                global_trend: Dict, max_layers: int = 6, 
                                filename_prefix: str = ''):
    """
    Visualize before and after correction for both real and imaginary signals.
    
    Creates comparison plots showing:
    - Original data with linear regression line
    - Corrected data with horizontal reference line
    
    Args:
        df_original: Original uncorrected DataFrame
        df_corrected: Corrected DataFrame
        global_trend: Dictionary with regression parameters
        max_layers: Maximum number of layers to display
        filename_prefix: Prefix for output filenames
    """
    if global_trend is None:
        print("❌ Global trend not available for visualization!")
        return
    
    layers = sorted(df_original['layer_index'].unique())
    
    # Exclude Layers 1 and 2 (not needed for visualization)
    layers = [layer for layer in layers if layer not in [1, 2]]
    
    # Select a subset of layers for visualization
    if len(layers) > max_layers:
        step = len(layers) // max_layers
        selected_layers = layers[::step][:max_layers]
        print(f"📊 Displaying {len(selected_layers)} layers out of {len(layers)} total "
              f"(excluding Layers 1 and 2): {selected_layers}")
    else:
        selected_layers = layers
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    geometry_order = ['CB', 'DB1', 'DB2', 'DB3', 'DB4', 'XCT']
    
    # Extract global trend parameters for reference lines
    global_real_slope = global_trend['global_real_slope']
    global_imag_slope = global_trend['global_imag_slope']
    real_intercept = global_trend['real_intercept']
    imag_intercept = global_trend['imag_intercept']
    
    # Plot for REAL signal
    fig, axes = plt.subplots(len(selected_layers), 2, figsize=(20, 6 * len(selected_layers)))
    if len(selected_layers) == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer in enumerate(selected_layers):
        # Plot original REAL
        ax_real = axes[i, 0]
        layer_orig = df_original[df_original['layer_index'] == layer]
        
        for j, geom in enumerate(geometry_order):
            geom_data = layer_orig[layer_orig['geometry_name'] == geom]
            if len(geom_data) > 0:
                ax_real.scatter(geom_data['x1'], geom_data['real'], 
                               color=colors[j], alpha=0.8, s=40, marker='o')
        
        # Global linear regression line
        x_range = np.linspace(layer_orig['x1'].min(), layer_orig['x1'].max(), 100)
        trend_line = global_real_slope * x_range + real_intercept
        ax_real.plot(x_range, trend_line, 'k--', alpha=0.9, linewidth=4, 
                    label='Linear regression')
        
        ax_real.set_title(f'Layer {layer} - REAL (Before correction)', 
                         fontsize=16, fontweight='bold', pad=20)
        ax_real.set_xlabel('x1 Position', fontsize=14)
        ax_real.set_ylabel('Real Signal', fontsize=14)
        ax_real.grid(True, alpha=0.3)
        ax_real.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot corrected REAL
        ax_real_corr = axes[i, 1]
        layer_corr = df_corrected[df_corrected['layer_index'] == layer]
        
        for j, geom in enumerate(geometry_order):
            geom_data = layer_corr[layer_corr['geometry_name'] == geom]
            if len(geom_data) > 0:
                ax_real_corr.scatter(geom_data['x1'], geom_data['real'], 
                                    color=colors[j], alpha=0.8, s=40, marker='o')
        
        # Horizontal line to show trend has been removed
        ax_real_corr.axhline(y=layer_corr['real'].mean(), color='black', 
                            linestyle='--', alpha=0.9, linewidth=4)
        
        ax_real_corr.set_title(f'Layer {layer} - REAL (After correction)', 
                              fontsize=16, fontweight='bold', pad=20)
        ax_real_corr.set_xlabel('x1 Position', fontsize=14)
        ax_real_corr.set_ylabel('Real Signal', fontsize=14)
        ax_real_corr.grid(True, alpha=0.3)
        ax_real_corr.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle('Recoater effect correction LINEAR REGRESSION (Stable layers 90-100) - REAL Signal', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Global legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=geometry_order[i]) 
                      for i in range(len(geometry_order))]
    legend_elements.append(Patch(facecolor='black', linestyle='--', label='Linear regression'))
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(geometry_order) + 1, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f'{filename_prefix}recoater_correction_REAL.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot for IMAGINARY signal
    fig, axes = plt.subplots(len(selected_layers), 2, figsize=(20, 6 * len(selected_layers)))
    if len(selected_layers) == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer in enumerate(selected_layers):
        # Plot original IMAGINARY
        ax_imag = axes[i, 0]
        layer_orig = df_original[df_original['layer_index'] == layer]
        
        for j, geom in enumerate(geometry_order):
            geom_data = layer_orig[layer_orig['geometry_name'] == geom]
            if len(geom_data) > 0:
                ax_imag.scatter(geom_data['x1'], geom_data['imag'], 
                               color=colors[j], alpha=0.8, s=40, marker='o')
        
        # Global linear regression line
        x_range = np.linspace(layer_orig['x1'].min(), layer_orig['x1'].max(), 100)
        trend_line = global_imag_slope * x_range + imag_intercept
        ax_imag.plot(x_range, trend_line, 'k--', alpha=0.9, linewidth=4, 
                    label='Linear regression')
        
        ax_imag.set_title(f'Layer {layer} - IMAGINARY (Before correction)', 
                         fontsize=16, fontweight='bold', pad=20)
        ax_imag.set_xlabel('x1 Position', fontsize=14)
        ax_imag.set_ylabel('Imaginary Signal', fontsize=14)
        ax_imag.grid(True, alpha=0.3)
        ax_imag.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot corrected IMAGINARY
        ax_imag_corr = axes[i, 1]
        layer_corr = df_corrected[df_corrected['layer_index'] == layer]
        
        for j, geom in enumerate(geometry_order):
            geom_data = layer_corr[layer_corr['geometry_name'] == geom]
            if len(geom_data) > 0:
                ax_imag_corr.scatter(geom_data['x1'], geom_data['imag'], 
                                    color=colors[j], alpha=0.8, s=40, marker='o')
        
        # Horizontal line to show trend has been removed
        ax_imag_corr.axhline(y=layer_corr['imag'].mean(), color='black', 
                            linestyle='--', alpha=0.9, linewidth=4)
        
        ax_imag_corr.set_title(f'Layer {layer} - IMAGINARY (After correction)', 
                              fontsize=16, fontweight='bold', pad=20)
        ax_imag_corr.set_xlabel('x1 Position', fontsize=14)
        ax_imag_corr.set_ylabel('Imaginary Signal', fontsize=14)
        ax_imag_corr.grid(True, alpha=0.3)
        ax_imag_corr.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle('Recoater effect correction LINEAR REGRESSION (Stable layers 90-100) - IMAGINARY Signal', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Global legend
    legend_elements = [Patch(facecolor=colors[i], label=geometry_order[i]) 
                      for i in range(len(geometry_order))]
    legend_elements.append(Patch(facecolor='black', linestyle='--', label='Linear regression'))
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(geometry_order) + 1, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f'{filename_prefix}recoater_correction_IMAGINARY.png', dpi=300, bbox_inches='tight')
    plt.close()


def process_channel(input_path: str, output_path: str, channel_name: str, 
                    stable_layers: List[int] = None):
    """
    Process a single channel with global recoater effect correction.
    
    Args:
        input_path: Path to input Excel file
        output_path: Path for output Excel file
        channel_name: Name of the channel (for logging)
        stable_layers: List of stable layer indices for trend calculation
    """
    print(f"\n{'=' * 60}")
    print(f"🔄 PROCESSING {channel_name}")
    print(f"{'=' * 60}")
    
    print(f"📂 Loading data: {input_path}")
    df = pd.read_excel(input_path)
    
    # Analyze recoater effect using global trend from stable layers
    global_trend = analyze_recoater_effect_global(df, stable_layers)
    
    if global_trend is None:
        print(f"❌ Cannot calculate global trend for {channel_name}")
        return
    
    # Correct recoater effect using global trend
    df_corrected = correct_recoater_effect_global(df, global_trend)
    
    # Visualize results (subset of layers) - save plots
    import os
    from pathlib import Path
    output_dir = Path(output_path).parent
    os.chdir(output_dir)  # Change directory to save plots
    
    visualize_correction_global(df, df_corrected, global_trend, max_layers=6, 
                               filename_prefix=f'{channel_name}_')
    
    # Restore original directory
    os.chdir('/Users/gianalbertocoldani/Desktop/Detrendare')
    
    # Save corrected data
    print(f"\n💾 Saving corrected data: {output_path}")
    df_corrected.to_excel(output_path, index=False)
    print(f"✅ {channel_name} completed!")


def main():
    """
    Main function for global recoater effect correction.
    Processes both channels using a global trend calculated from stable layers.
    """
    print("🎯 RECOATER EFFECT CORRECTION - LINEAR REGRESSION (Stable layers 90-100)")
    print("=" * 60)
    
    # Define stable layers for global trend calculation
    stable_layers = list(range(90, 101))  # Layers 90-100
    print(f"📊 Stable layers used for global trend: {stable_layers}")
    
    # Paths for Channel 0 (BJ3)
    ch0_input = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ3/0_Boxes.filtered_improved_fallback.xlsx'
    ch0_output = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ3/0_Boxes.recoater_corrected_REGRESSION.xlsx'
    
    # Paths for Channel 1 (BJ3)
    ch1_input = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ3/1_Boxes.filtered_improved_fallback.xlsx'
    ch1_output = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ3/1_Boxes.recoater_corrected_REGRESSION.xlsx'
    
    # Process Channel 0
    process_channel(ch0_input, ch0_output, "CHANNEL_0", stable_layers)
    
    # Process Channel 1
    process_channel(ch1_input, ch1_output, "CHANNEL_1", stable_layers)
    
    print(f"\n{'=' * 60}")
    print("🎉 ALL CHANNELS HAVE BEEN CORRECTED SUCCESSFULLY!")
    print("=" * 60)
    print("📊 The correction now:")
    print("   • Calculates linear regression on all points from stable layers (90-100)")
    print("   • Uses all points from CB, DB1, DB2, DB3, DB4 for robustness")
    print("   • Applies this fixed trend to all layers")
    print("   • CB remains unchanged (reference point)")
    print("   • The further from CB, the larger the correction applied")
    print("   • Processes real and imaginary components separately")
    print("   • Processes both channels")
    print("   • Includes R² metrics to evaluate regression quality")


if __name__ == "__main__":
    main()
