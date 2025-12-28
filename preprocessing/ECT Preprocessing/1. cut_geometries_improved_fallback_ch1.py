#!/usr/bin/env python3
"""
================================================================================
ECT DATA PREPROCESSING - STEP 1: GEOMETRY CUTTING (Channel 1)
================================================================================
Purpose: ECT Data Preprocessing Pipeline

DESCRIPTION:
    This script performs the first preprocessing step for ECT data. It filters and cuts ECT measurement data by identifying
    regions of interest within each geometry-layer combination.
    
    NOTE: This is the Channel 1 version. The algorithm is identical to Channel 0,
    only the input/output paths differ.

ALGORITHM:
    1. For each geometry-layer combination, the script identifies two peaks in the
       signal (sorted by spatial position x1) that are spatially separated by at
       least 3 positions.
    
    2. The data points BETWEEN these two peaks (excluding the peaks themselves)
       are kept, as they represent the stable measurement region.
    
    3. If peak detection fails (insufficient points between peaks), a fallback
       strategy is applied using average cut sizes calculated from successful
       detections on the same geometry type.

GEOMETRY-SPECIFIC PARAMETERS:
    - CB, DB1, DB2, DB3, DB4: Minimum 7 points required between peaks
    - XCT: Minimum 3 points required between peaks
    
    Default fallback cut sizes:
    - CB, DB1, DB2, DB3, DB4: 8 points
    - XCT: 4 points

INPUT:
    Excel file containing ECT data with columns:
    - geometry_name: Name of the geometry (CB, DB1, DB2, DB3, DB4, XCT)
    - layer_index: Layer number
    - x1: Spatial position
    - real: Real component of the signal (used for peak detection)

OUTPUT:
    Filtered Excel file containing only the data points between detected peaks,
    sorted by geometry, layer, and spatial position.

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional


def infer_signal_column(df: pd.DataFrame, prefer_magnitude: bool = True) -> Tuple[str, Optional[pd.Series]]:
    """
    Determine which signal column to use for peak detection.
    
    Priority order:
    1. 'real' column if present (preferred for ECT data)
    2. First numeric column that isn't metadata
    3. Any numeric column as last resort
    
    Args:
        df: Input DataFrame containing signal data
        prefer_magnitude: Legacy parameter, kept for compatibility
        
    Returns:
        Tuple of (column_name, signal_values_series)
        
    Raises:
        ValueError: If no numeric columns are found
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    meta_like = {'datapoint_id', 'session_id', 'layer_index', 'channel', 'config_id'}

    # Always prefer 'real' column as specified for ECT data
    if 'real' in df.columns:
        return 'real', df['real'].astype(float)

    # Fallback: first numeric column that is not obviously metadata
    for col in numeric_cols:
        if col not in meta_like:
            return col, df[col].astype(float)

    # If everything fails, use the first numeric column
    if numeric_cols:
        col = numeric_cols[0]
        return col, df[col].astype(float)

    raise ValueError('No numeric columns found to determine signal.')


def filter_between_top_two_peaks(group: pd.DataFrame, signal_values: pd.Series) -> pd.DataFrame:
    """
    Keep rows strictly between the two highest signal values in spatial order.
    
    This function identifies two peaks in the signal that are spatially separated
    by at least 3 positions, then returns only the data points between them
    (excluding the peaks themselves).
    
    Args:
        group: DataFrame containing data for a single geometry-layer combination
        signal_values: Series containing signal values for peak detection
        
    Returns:
        DataFrame containing only the rows between the two detected peaks
    """
    if len(group) < 3:
        return group.iloc[0:0]  # Not enough points to keep anything strictly inside

    # Sort by x1 (spatial order)
    if 'x1' not in group.columns:
        return group.iloc[0:0]  # Need x1 for spatial ordering
    
    group_sorted = group.sort_values('x1').reset_index(drop=True)
    
    # Get signal values for this sorted group using the original indices
    original_indices = group.sort_values('x1').index
    local_signal = signal_values.loc[original_indices].reset_index(drop=True)
    if local_signal.isna().all():
        return group.iloc[0:0]

    # Find the two highest values (least negative for real signal)
    # Ensure they are spatially separated by at least 3 positions
    sorted_values = local_signal.sort_values(ascending=False)
    
    # Find two peaks that are spatially separated by at least 3 positions
    peak1_idx = None
    peak2_idx = None
    
    for i, (idx, val) in enumerate(sorted_values.items()):
        if peak1_idx is None:
            peak1_idx = idx
        elif peak2_idx is None:
            # Check if this peak is spatially separated from the first one
            pos1 = group_sorted.index.get_loc(peak1_idx)
            pos2 = group_sorted.index.get_loc(idx)
            if abs(pos1 - pos2) >= 3:  # At least 3 positions apart
                peak2_idx = idx
                break
    
    if peak1_idx is None or peak2_idx is None:
        return group.iloc[0:0]
    
    # Find positions in the sorted order
    peak1_pos = group_sorted.index.get_loc(peak1_idx)
    peak2_pos = group_sorted.index.get_loc(peak2_idx)
    
    # Determine start and end positions (include points immediately after/before peaks)
    start_pos = min(peak1_pos, peak2_pos) + 1  # Include point after first peak
    end_pos = max(peak1_pos, peak2_pos) - 1    # Include point before second peak
    
    if start_pos > end_pos:
        return group.iloc[0:0]
    
    # Get the rows to keep from the sorted group
    kept_rows = group_sorted.iloc[start_pos:end_pos + 1]
    
    return kept_rows


def find_peaks(group: pd.DataFrame, signal_values: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    """
    Find two peaks that are spatially separated by at least 3 positions.
    
    Args:
        group: DataFrame containing data for a single geometry-layer combination
        signal_values: Series containing signal values for peak detection
        
    Returns:
        Tuple of (peak1_idx, peak2_idx) or (None, None) if not found
    """
    if len(group) < 3:
        return None, None

    # Sort by x1 (spatial order)
    if 'x1' not in group.columns:
        return None, None
    
    group_sorted = group.sort_values('x1').reset_index(drop=True)
    
    # Get signal values for this sorted group using the original indices
    original_indices = group.sort_values('x1').index
    local_signal = signal_values.loc[original_indices].reset_index(drop=True)
    if local_signal.isna().all():
        return None, None

    # Find the two highest values (least negative for real signal)
    sorted_values = local_signal.sort_values(ascending=False)
    
    # Find two peaks that are spatially separated by at least 3 positions
    peak1_idx = None
    peak2_idx = None
    
    for i, (idx, val) in enumerate(sorted_values.items()):
        if peak1_idx is None:
            peak1_idx = idx
        elif peak2_idx is None:
            # Check if this peak is spatially separated from the first one
            pos1 = group_sorted.index.get_loc(peak1_idx)
            pos2 = group_sorted.index.get_loc(idx)
            if abs(pos1 - pos2) >= 3:  # At least 3 positions apart
                peak2_idx = idx
                break
    
    return peak1_idx, peak2_idx


def filter_by_fixed_range(group: pd.DataFrame, signal_values: pd.Series, 
                          start_pos: int, end_pos: int) -> pd.DataFrame:
    """
    Keep points in a fixed range but exclude the peak closest to an edge.
    
    This fallback function is used when standard peak detection fails.
    It selects continuous points while avoiding edge peaks.
    
    Args:
        group: DataFrame containing data for a single geometry-layer combination
        signal_values: Series containing signal values for peak detection
        start_pos: Starting position index
        end_pos: Ending position index
        
    Returns:
        DataFrame containing filtered rows
    """
    if len(group) < 1:
        return group.iloc[0:0]
    
    # Sort by x1 (spatial order)
    if 'x1' not in group.columns:
        return group.iloc[0:0]
    
    group_sorted = group.sort_values('x1').reset_index(drop=True)
    
    # Find peaks to determine which one is closer to edge
    peak1_idx, peak2_idx = find_peaks(group, signal_values)
    
    # Adjust positions to be within bounds
    start_pos = max(0, start_pos)
    end_pos = min(len(group_sorted) - 1, end_pos)
    
    if start_pos > end_pos:
        return group.iloc[0:0]
    
    # Determine which peak is closer to an edge
    edge_peak_idx = None
    edge_peak_pos = None
    
    if peak1_idx is not None and peak2_idx is not None:
        peak1_pos = group_sorted.index.get_loc(peak1_idx)
        peak2_pos = group_sorted.index.get_loc(peak2_idx)
        
        # Calculate distances to edges
        dist_peak1_to_left = peak1_pos
        dist_peak1_to_right = len(group_sorted) - 1 - peak1_pos
        dist_peak2_to_left = peak2_pos
        dist_peak2_to_right = len(group_sorted) - 1 - peak2_pos
        
        min_dist_peak1 = min(dist_peak1_to_left, dist_peak1_to_right)
        min_dist_peak2 = min(dist_peak2_to_left, dist_peak2_to_right)
        
        # Choose the peak closer to an edge
        if min_dist_peak1 < min_dist_peak2:
            edge_peak_idx = peak1_idx
            edge_peak_pos = peak1_pos
        else:
            edge_peak_idx = peak2_idx
            edge_peak_pos = peak2_pos
    elif peak1_idx is not None:
        edge_peak_idx = peak1_idx
        edge_peak_pos = group_sorted.index.get_loc(peak1_idx)
    elif peak2_idx is not None:
        edge_peak_idx = peak2_idx
        edge_peak_pos = group_sorted.index.get_loc(peak2_idx)
    
    # If we found an edge peak, select continuous points excluding it
    if edge_peak_idx is not None and edge_peak_pos is not None:
        # Determine if peak is closer to left or right edge
        dist_to_left = edge_peak_pos
        dist_to_right = len(group_sorted) - 1 - edge_peak_pos
        
        # Determine target number of points based on original range
        target_points = end_pos - start_pos + 1
        
        if dist_to_left < dist_to_right:
            # Peak is closer to left edge, select points after it
            new_start = edge_peak_pos + 1
            new_end = min(new_start + target_points - 1, len(group_sorted) - 1)
        else:
            # Peak is closer to right edge, select points before it
            new_end = edge_peak_pos - 1
            new_start = max(new_end - target_points + 1, 0)
        
        # Ensure we have the target number of points if possible
        if new_end - new_start + 1 >= target_points:
            new_end = new_start + target_points - 1
        
        if new_start <= new_end and new_start >= 0 and new_end < len(group_sorted):
            return group_sorted.iloc[new_start:new_end + 1]
    
    # Fallback to original range if no edge peak found
    return group_sorted.iloc[start_pos:end_pos + 1]


def calculate_average_cut_size(df: pd.DataFrame, geometry_col: str = 'geometry_name', 
                               layer_col: str = 'layer_index') -> dict:
    """
    Calculate the average cut size for each geometry based on successful peak detection.
    
    Peak detection is considered successful if:
    - CB, DB1, DB2, DB3, DB4: cut between peaks (peaks excluded) >= 7 points
    - XCT: cut between peaks (peaks excluded) >= 3 points
    
    Args:
        df: Input DataFrame
        geometry_col: Name of the geometry column
        layer_col: Name of the layer column
        
    Returns:
        Dictionary mapping geometry names to average cut sizes
    """
    print("🔍 Calculating average cut sizes per geometry...")
    
    geometries = df[geometry_col].unique()
    avg_sizes = {}
    
    # Define minimum cut sizes for successful peak detection
    min_cut_sizes = {
        'CB': 7, 'DB1': 7, 'DB2': 7, 'DB3': 7, 'DB4': 7, 'XCT': 3
    }
    
    for geom in geometries:
        geom_data = df[df[geometry_col] == geom]
        successful_cuts = []
        
        for layer in geom_data[layer_col].unique():
            layer_geom_data = geom_data[geom_data[layer_col] == layer]
            
            if len(layer_geom_data) >= 3:
                # Try peak detection
                signal_values = layer_geom_data['real']
                peak1_idx, peak2_idx = find_peaks(layer_geom_data, signal_values)
                
                if peak1_idx is not None and peak2_idx is not None:
                    # Calculate cut size between peaks (peaks excluded)
                    group_sorted = layer_geom_data.sort_values('x1').reset_index(drop=True)
                    peak1_pos = group_sorted.index.get_loc(peak1_idx)
                    peak2_pos = group_sorted.index.get_loc(peak2_idx)
                    
                    # Cut size between peaks (excluding peaks themselves)
                    cut_size = abs(peak2_pos - peak1_pos) - 1
                    
                    # Check if cut size meets minimum requirement
                    min_required = min_cut_sizes.get(geom, 7)
                    if cut_size >= min_required:
                        successful_cuts.append(cut_size)
        
        if successful_cuts:
            avg_size = int(round(sum(successful_cuts) / len(successful_cuts)))
            avg_sizes[geom] = avg_size
            print(f"  {geom}: {len(successful_cuts)} successful cuts "
                  f"(>={min_cut_sizes.get(geom, 7)} points), average size: {avg_size} points")
        else:
            # Fallback to default sizes if no successful cuts found
            default_sizes = {'CB': 8, 'DB1': 8, 'DB2': 8, 'DB3': 8, 'DB4': 8, 'XCT': 4}
            avg_sizes[geom] = default_sizes.get(geom, 8)
            print(f"  {geom}: no successful cuts, using default size: {avg_sizes[geom]} points")
    
    return avg_sizes


def process_full_dataset_improved_fallback(input_path: str, output_path: Optional[str] = None,
                                           geometry_col: str = 'geometry_name', 
                                           layer_col: str = 'layer_index',
                                           prefer_magnitude: bool = True) -> pd.DataFrame:
    """
    Process the full dataset with improved fallback logic based on analysis of good cases.
    
    The fallback ranges are based on analysis of successful peak detections:
    - CB: positions 5-12 (8 points)
    - DB1: positions 5-12 (8 points)  
    - DB2: positions 5-12 (8 points)
    - DB3: positions 6-13 (8 points)
    - DB4: positions 5-12 (8 points)
    - XCT: positions 6-10 (5 points)
    
    Args:
        input_path: Path to the input Excel file
        output_path: Path for the output file (auto-generated if None)
        geometry_col: Name of the geometry column
        layer_col: Name of the layer column
        prefer_magnitude: Legacy parameter for signal selection
        
    Returns:
        Filtered DataFrame
    """
    input_path = str(Path(input_path).expanduser())
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_name(p.stem + '.filtered_improved_fallback.xlsx'))

    print(f"📂 Loading complete dataset: {input_path}")
    
    # Read the Excel file
    try:
        xls = pd.ExcelFile(input_path)
        sheet = 'Data' if 'Data' in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        raise RuntimeError(f'Failed to read Excel: {e}')

    if geometry_col not in df.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not found in Excel.")
    if layer_col not in df.columns:
        raise ValueError(f"Layer column '{layer_col}' not found in Excel.")

    # Determine signal
    signal_name, signal_values = infer_signal_column(df, prefer_magnitude=prefer_magnitude)
    print(f"📊 Using signal: {signal_name}")

    # Calculate average cut sizes based on successful peak detection
    avg_cut_sizes = calculate_average_cut_size(df, geometry_col, layer_col)
    
    print(f"\n📊 Calculated average sizes for fallback:")
    for geom, size in avg_cut_sizes.items():
        print(f"  {geom}: {size} points")

    # Get unique geometries and layers
    geometries = sorted(df[geometry_col].unique())
    layers = sorted(df[layer_col].unique())
    print(f"📊 Geometries: {geometries}")
    print(f"📊 Layer range: {layers[0]} - {layers[-1]} ({len(layers)} total layers)")

    # Get unique combinations of geometry and layer
    unique_combinations = df[[geometry_col, layer_col]].drop_duplicates().sort_values(
        [geometry_col, layer_col])
    total_combinations = len(unique_combinations)
    print(f"📈 Geometry-layer combinations: {total_combinations}")
    
    kept_parts = []
    fallback_count = 0
    
    with tqdm(total=total_combinations, desc="🔄 Processing geometry-layers", unit="combo") as pbar:
        for _, row in unique_combinations.iterrows():
            geom = row[geometry_col]
            layer = row[layer_col]
            
            # Filter data for this geometry-layer combination
            mask = (df[geometry_col] == geom) & (df[layer_col] == layer)
            group_data = df[mask].copy()
            
            if len(group_data) > 0:
                # Try standard logic first
                filtered_standard = filter_between_top_two_peaks(group_data, signal_values)
                
                # Check if we need fallback
                if geom in ['CB', 'DB1', 'DB2', 'DB3', 'DB4']:
                    min_points = 7
                elif geom == 'XCT':
                    min_points = 3
                else:
                    min_points = 5
                
                if len(filtered_standard) >= min_points:
                    filtered_group = filtered_standard
                else:
                    # Use fallback with average cut size
                    if geom in avg_cut_sizes:
                        target_size = avg_cut_sizes[geom]
                        # Create a range centered in the middle of available points
                        total_points = len(group_data)
                        if total_points >= target_size:
                            start_pos = max(0, (total_points - target_size) // 2)
                            end_pos = min(total_points - 1, start_pos + target_size - 1)
                            filtered_group = filter_by_fixed_range(
                                group_data, signal_values, start_pos, end_pos)
                            fallback_count += 1
                        else:
                            filtered_group = filtered_standard
                    else:
                        filtered_group = filtered_standard
                
                if not filtered_group.empty:
                    kept_parts.append(filtered_group)
            
            pbar.update(1)
            
            # Update progress info
            if pbar.n % 1000 == 0:
                pbar.set_postfix({
                    'Processed': pbar.n,
                    'Kept': len(kept_parts),
                    'Fallback': fallback_count,
                    'Geometry': geom,
                    'Layer': layer
                })

    # Combine all kept data
    if kept_parts:
        print(f"\n📊 Combining {len(kept_parts)} kept groups...")
        filtered = pd.concat(kept_parts, axis=0, ignore_index=True)
        
        # Sort by geometry, layer, and x1
        sort_cols = [geometry_col, layer_col]
        if 'x1' in filtered.columns:
            sort_cols.append('x1')
        filtered = filtered.sort_values(sort_cols).reset_index(drop=True)
    else:
        filtered = df.iloc[0:0]

    # Save Excel
    output_path = str(Path(output_path).expanduser())
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving filtered Excel: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        filtered.to_excel(writer, sheet_name='Data', index=False)

    print(f"✅ Completed!")
    print(f"📊 Original points: {len(df):,}")
    print(f"📊 Kept points: {len(filtered):,}")
    print(f"📊 Reduction: {len(filtered)/len(df)*100:.1f}%")
    print(f"📊 Combinations using fallback: {fallback_count:,}")
    
    # Statistics per geometry
    print(f"\n📈 Points kept per geometry:")
    for geom in geometries:
        geom_count = len(filtered[filtered[geometry_col] == geom])
        print(f"  {geom}: {geom_count:,} points")
    
    return filtered


def main():
    """
    Main entry point for Channel 1 geometry cutting.
    """
    # Single file processing for BJ4 - Channel 1
    input_path = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ4/geometrie_excel_BJ4/1_All_Bj4.xlsx'
    output_path = '/Users/gianalbertocoldani/Desktop/Detrendare/0.Geometrie tagliate/0.Geometrie tagliate_BJ4/geometrie_excel_BJ4/1_All_Bj4.filtered_improved_fallback.xlsx'
    
    filtered_df = process_full_dataset_improved_fallback(input_path, output_path)
    
    print(f"\n🎯 Improved fallback logic applied!")
    print(f"📁 Generated file: {output_path}")


if __name__ == '__main__':
    main()
