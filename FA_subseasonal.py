
# -*- coding: utf-8 -*-
"""
Subseasonal Frontal Ablation Calculation 
==========================================================

This module calculates frontal ablation rates for glacier termini using the
multiple-fluxgate method. 

The workflow consists of three steps:
    1. FA_flowlines: Calculate variables along individual flowlines
    2. FA_fluxgates: Combine adjacent flowlines into fluxgates
    3. FA_interval: Aggregate fluxgates to interval (40-90 days) totals

Author: Lautaro Clavero
Email: lclavero@mendoza-conicet.gob.ar
"""

import time
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from osgeo import gdal
from FA_subseasonal_utils import (
    list_files_FA,
    split_lines_by_id,
    get_velocity,
    get_point_value,
    sample_raster_at_points,
    nmad_error,
    calculate_subseasonal_thickness,
    clear_all_caches 
)


def FA_flowlines(interval_list: list, flowline_path: str, H_zero_path: str, 
                 dhdt_map_sub: dict, shapes_stables: list,
                 terminus_boundary: int = 25, sliding_factor: float = 0.997) -> pd.DataFrame:
    """
    Calculate frontal ablation variables for each flowline and interval.
    
    Computes along each flowline:
        - Ut: Ice velocity at terminus (m/day)
        - dL/dt: Terminus displacement rate (m/day, positive = advance)
        - Uc: Frontal ablation rate = Ut - dL/dt (m/day)
        - Ht: Ice thickness at terminus (m)
    
    Args:
        interval_list: List of interval dictionaries from list_files_FA()
        flowline_path: Path to shapefile with flowline geometries
        H_zero_path: Path to reference thickness raster (2012)
        dhdt_map_sub: Dict mapping time periods to {'dhdt': value, 'err': value}
        shapes_stables: List of stable bedrock polygon paths for error estimation
        terminus_boundary: Number of points behind terminus for averaging (default: 25)
        sliding_factor: Factor to reduce velocity at terminus (default: 0.997)
    
    Returns:
        DataFrame with columns: Start_Date, End_Date, Interval_Days, ID, Ut, err_Ut,
        dL/dt, err_dL/dt, Uc, err_Uc, Ht, err_Ht, XY_front_start, XY_front_end, px_size_mask
    
    Note:
        Flowlines are filtered to ensure complete data (velocity and thickness available).
        Invalid flowlines are silently skipped.
    """
    
    flowlines = split_lines_by_id(flowline_path)
    dfs = [] 
    zero_date = '20120421'  # Reference date for initial thickness 

    # Caches for performance
    err_ut_cache = {}    
    mask_gt_cache = {}
    
    for interval_dict in tqdm(interval_list, desc="Processing intervals"):
        
        # Load velocity grids for this interval
        raw_vel = get_velocity(interval_dict['raw_vels'], interval_dict['coefficients'])
        final_vel = get_velocity(interval_dict['final_vels'], interval_dict['coefficients'])
        
        # Calculate Ut error once per interval (using stable bedrock)
        ut_key = interval_dict['raw_vels'][0]
        if ut_key not in err_ut_cache:
            err_ut_cache[ut_key] = nmad_error(interval_dict['raw_vels'][0], raw_vel, shapes_stables) / 365
        err_Ut = err_ut_cache[ut_key]
        
        # Cache mask geotransforms
        mask_ini_path = interval_dict['mask_ini']
        mask_fin_path = interval_dict['mask_fin']
        
        if mask_ini_path not in mask_gt_cache:
            mask_gt_cache[mask_ini_path] = gdal.Open(mask_ini_path).GetGeoTransform()
        if mask_fin_path not in mask_gt_cache:
            mask_gt_cache[mask_fin_path] = gdal.Open(mask_fin_path).GetGeoTransform()
        
        gt_ini = mask_gt_cache[mask_ini_path]
        pixelSizeX, pixelSizeY = gt_ini[1], -gt_ini[5]
        err_dL_dt_base = sqrt(pixelSizeX**2 + pixelSizeY**2)  # Mask grid resolution error
        
        # Parse dates
        datetime_start = datetime.strptime(interval_dict['date_ini'], '%Y%m%d')
        datetime_end = datetime.strptime(interval_dict['date_fin'], '%Y%m%d')
        delta_T = interval_dict['interval_days']
        datetime_center = datetime_start + timedelta(days=delta_T // 2)
        
        for fl in flowlines:
            
            line_A = fl.copy()
            line_B = fl.copy()
            
            ID = int(line_A['LINE_ID'].mean())
            
            # Extract velocity and mask values along flowlines
            vel = get_point_value(line_A, final_vel, interval_dict['final_vels'][0])
            mask_start = sample_raster_at_points(line_A, interval_dict['mask_ini'])
            mask_end = sample_raster_at_points(line_B, interval_dict['mask_fin'])
            
            # Front detection: find where mask value != 1 (non-glacier)
            if len(mask_start.loc[mask_start.value != 1]) == 0 or len(mask_end.loc[mask_end.value != 1]) == 0:
                continue  # No terminus detected in this interval
            
            a = mask_start.loc[mask_start.value != 1].index.values[0] - 1
            b = mask_end.loc[mask_end.value != 1].index.values[0] - 1
            c = a - terminus_boundary
            d = b - terminus_boundary
            
            n = len(vel)
            def clamp(x): return min(max(int(x), 0), n - 1)
            
            i1, i2, i3, i4 = clamp(c), clamp(a), clamp(d), clamp(b)
            
            # Front coordinates
            x_front_start, y_front_start = vel.iloc[i2].X, vel.iloc[i2].Y
            x_front_end, y_front_end = vel.iloc[i4].X, vel.iloc[i4].Y
            
            dx = sqrt((x_front_end - x_front_start)**2 + (y_front_end - y_front_start)**2)
            
            # Reference thickness at zero date
            H_start = sample_raster_at_points(line_A, H_zero_path)
            
            # Calculate dL/dt (terminus displacement rate)
            if i2 < i4:  # Glacier advance
                front_rows = vel.iloc[i1:(i2 + 1)]
                idx_start, idx_end = i1, i2
                dL_dt = dx / delta_T
            else:  # Glacier retreat
                front_rows = vel.iloc[i3:(i4 + 1)]
                idx_start, idx_end = i3, i4
                dL_dt = -dx / delta_T
            
            err_dL_dt = err_dL_dt_base / delta_T

            # Validate data completeness
            if len(front_rows) == 0:
                continue
            if np.all(np.isnan(front_rows.velocity)):
                continue
            if idx_end <= idx_start:
                continue 
            
            # Calculate thickness at target date
            H_rows, err_H_rows = calculate_subseasonal_thickness(
                datetime_center, zero_date, H_start, dhdt_map_sub, idx_start, idx_end
            )
            
            # Store results
            dfs.append({
                'Start_Date': datetime_start,
                'End_Date': datetime_end,
                'Interval_Days': delta_T,
                'ID': ID,
                'Ut': (np.nanmean(front_rows.velocity) / 365) * sliding_factor,
                'err_Ut': err_Ut,
                'dL/dt': dL_dt,
                'err_dL/dt': err_dL_dt,
                'Uc': (np.nanmean(front_rows.velocity) / 365) * sliding_factor - dL_dt,
                'err_Uc': sqrt(err_Ut**2 + err_dL_dt**2),
                'Ht': H_rows.mean(),
                'err_Ht': np.sqrt(np.mean(err_H_rows**2)),
                'XY_front_start': (x_front_start, y_front_start),
                'XY_front_end': (x_front_end, y_front_end),
                'px_size_mask': pixelSizeX
            })

    return pd.DataFrame(dfs)


def FA_fluxgates(flowline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frontal ablation for each fluxgate (area between two flowlines).
    
    A fluxgate is defined by two adjacent flowlines. For each fluxgate, calculates:
        - Uc: Mean frontal ablation rate (m/day)
        - Hc: Mean ice thickness (m)
        - Wi: Fluxgate width (m)
        - Qc: Frontal ablation volume flux = Uc * Hc * Wi (m³/day)
    
    Args:
        flowline_df: DataFrame from FA_flowlines() with columns:
                     Start_Date, ID, Uc, err_Uc, Ht, err_Ht, XY_front_start, 
                     XY_front_end, px_size_mask
    
    Returns:
        DataFrame with fluxgate-level results including Qc (m³/day) and errors
    """
    results = []
    
    for start_date, interval in flowline_df.groupby('Start_Date'):
        interval = interval.sort_values('ID').reset_index(drop=True)
        
        for i in range(1, len(interval)):
            j = i - 1
            
            # Calculate widths at start and end of interval
            xi_s, yi_s = interval.iloc[i].XY_front_start
            xj_s, yj_s = interval.iloc[j].XY_front_start
            wa = sqrt((xi_s - xj_s)**2 + (yi_s - yj_s)**2)
            
            xi_e, yi_e = interval.iloc[i].XY_front_end
            xj_e, yj_e = interval.iloc[j].XY_front_end
            wb = sqrt((xi_e - xj_e)**2 + (yi_e - yj_e)**2)
            
            # Average width and error
            wi = (wa + wb) / 2
            err_wi = sqrt(((abs(wa - wb) / 2)**2) + ((interval.iloc[j].px_size_mask * 0.5)**2))
            
            # Average values between two flowlines
            Uc = (interval.iloc[i].Uc + interval.iloc[j].Uc) / 2          
            err_Uc = sqrt(0.5 * (interval.iloc[i].err_Uc)**2)
            
            Hc = (interval.iloc[i].Ht + interval.iloc[j].Ht) / 2
            err_Hc = sqrt(0.5 * (interval.iloc[i].err_Ht)**2)
            
            # Frontal ablation volume flux
            Qc = Uc * Hc * wi                                     
            err_Qc = sqrt((err_Uc * Qc / Uc)**2 + (err_Hc * Qc / Hc)**2 + (err_wi * Qc / wi)**2)
            
            results.append({
                'Start': interval.iloc[j].Start_Date,
                'End': interval.iloc[j].End_Date,
                'Interval_Days': interval.iloc[j].Interval_Days,
                'Gate ID': f"{int(interval.iloc[j].ID):02d}{int(interval.iloc[i].ID):02d}",
                'Uc': Uc, 'err_Uc': err_Uc,
                'Hc': Hc, 'err_Hc': err_Hc,
                'Wi': wi, 'err_Wi': err_wi,
                'Qc': Qc, 'err_Qc': err_Qc
            })
    
    return pd.DataFrame(results).sort_values('Start').reset_index(drop=True)


def FA_interval(fluxgate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fluxgates to interval (monthly) level.
    
    Sums Qc across all fluxgates for each time interval to get total
    frontal ablation rate and ice loss volume.
    
    Args:
        fluxgate_df: DataFrame from FA_fluxgates() with columns:
                     Start, End, Interval_Days, Qc, err_Qc
    
    Returns:
        DataFrame with aggregated results:
            - Frontal_ablation_m3d: Total ablation rate (m³/day)
            - Ice_Loss_m3: Total ice loss volume (m³)
            - Ice_Loss_Mm3: Ice loss in million m³
    """
    results = []
    
    for start_date, interval in fluxgate_df.groupby('Start'):
        Qc_sum = interval['Qc'].sum()
        err_Qc_sum = sqrt((interval['err_Qc']**2).sum())
        days = interval.iloc[0]['Interval_Days']
        
        results.append({
            'Start': start_date,
            'End': interval.iloc[0]['End'],
            'Interval_Days': days,
            'Frontal_ablation_m3d': Qc_sum,
            'Err_Frontal_ablation': err_Qc_sum,
            'Ice_Loss_m3': Qc_sum * days,
            'Err_Ice_Loss_m3': err_Qc_sum * days,
            'Ice_Loss_Mm3': Qc_sum * days / 1e6,
            'Err_Ice_Loss_Mm3': err_Qc_sum * days / 1e6
        })
    
    return pd.DataFrame(results).sort_values('Start').reset_index(drop=True)


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ==================== INPUT PATHS ====================

    SCRIPT_DIR = Path(__file__).parent
    
    # Data directory
    DATA_DIR = SCRIPT_DIR / "Input_Data_Manso"
    
    # Data folder containing masks, raw velocities, and final velocities
    DATA_FOLDER = DATA_DIR / "data_Manso_subseasonal"
    
    # Reference thickness raster (Zorzut et al. 2020)
    THICKNESS_2012 = DATA_DIR / "thickness" / "thickness_2012_Manso.tif"
    
    # Flowline shapefile (MultiPoint geometry with LINE_ID attribute)
    FLOWLINE_SHP = DATA_DIR / "geometry" / "Flowlines.shp"
    
    # Stable bedrock polygons for error estimation
    STABLE_BEDROCK_SHAPES = [
        DATA_DIR / "geometry" / "stable_zones" / "stable_1.shp",
        DATA_DIR / "geometry" / "stable_zones" / "stable_2.shp",
        DATA_DIR / "geometry" / "stable_zones" / "stable_3.shp",
        DATA_DIR / "geometry" / "stable_zones" / "stable_4.shp"
    ]
    
    # Convert to strings for compatibility with existing functions
    STABLE_BEDROCK_SHAPES = [str(p) for p in STABLE_BEDROCK_SHAPES]
    
    # Annual dh/dt rates for thickness propagation (from Pleiades DEMs differencing)
    DHDT_MAP = {
        (datetime(2012, 4, 21), datetime(2017, 3, 6)):   {"dhdt": -6.78, "err": 0.21},
        (datetime(2017, 3, 6), datetime(2018, 3, 5)):    {"dhdt": -9.88, "err": 0.28},
        (datetime(2018, 3, 5), datetime(2019, 3, 11)):   {"dhdt": -9.64, "err": 0.37},
        (datetime(2019, 3, 11), datetime(2020, 3, 2)):   {"dhdt": -5.95, "err": 0.34},
        (datetime(2020, 3, 2), datetime(2021, 3, 2)):    {"dhdt": -5.52, "err": 0.45},
        (datetime(2021, 3, 2), datetime(2022, 3, 12)):   {"dhdt": -8.34, "err": 0.63},
        (datetime(2022, 3, 12), datetime(2023, 3, 5)):   {"dhdt": -6.79, "err": 0.57},
        (datetime(2023, 3, 5), datetime(2024, 3, 4)):    {"dhdt": -7.31, "err": 0.36},
    }
    
    # Check if input directory exists
    if not DATA_FOLDER.exists():
        print(f"\nERROR: Data folder not found: {DATA_FOLDER}")
        print("Please ensure the data folder structure is correct.")
        print("Expected: Input_Data_Manso/data_Manso_subseasonal/")
        exit(1)
    
    # ==================== RUN CALCULATION ====================
    
    print("=" * 60)
    print("Subseasonal Frontal Ablation Calculation")
    print("=" * 60)
    
    # Generate interval list (40-90 day windows)
    start_time = time.time()
    print("\n1. Generating interval list...")
    interval_list = list_files_FA(str(DATA_FOLDER), min_days=40, max_days=90)
    print(f"   Found {len(interval_list)} valid intervals")
    
    # Calculate flowline variables
    print("\n2. Calculating flowline variables...")
    df_flow = FA_flowlines(
        interval_list, 
        str(FLOWLINE_SHP), 
        str(THICKNESS_2012), 
        DHDT_MAP, 
        STABLE_BEDROCK_SHAPES
    )
    flow_time = (time.time() - start_time) / 60
    print(f"   Flowlines ready. Time: {flow_time:.2f} minutes")
    print(f"   Flowline records: {len(df_flow)}")
    
    # Filter outliers (based on Manso Glacier characteristics)
    MAX_UC_MANSO_GLACIER = 2.0  # m/day - values above this are considered outliers
    
    print("\n3. Filtering outliers...")
    n_before = len(df_flow)
    df_flow = df_flow[df_flow['Ut'] > 0]
    df_flow = df_flow[df_flow['Uc'] > 0]
    df_flow = df_flow[df_flow['Uc'] < MAX_UC_MANSO_GLACIER]
    print(f"   Removed {n_before - len(df_flow)} outlier records")
    
    # Calculate fluxgates
    print("\n4. Calculating fluxgates...")
    df_flux = FA_fluxgates(df_flow)
    print(f"   Fluxgates: {len(df_flux)}")
    
    # Aggregate to interval level
    print("\n5. Aggregating to interval level...")
    df_FA = FA_interval(df_flux)
    print(f"   Intervals: {len(df_FA)}")
    
    # ==================== SAVE RESULTS ====================
    
    # Create output directory
    output_dir = SCRIPT_DIR / "output" / "subseasonal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to CSV
    df_flow.to_csv(output_dir / "flowlines_results.csv", index=False)
    df_flux.to_csv(output_dir / "fluxgates_results.csv", index=False)
    df_FA.to_csv(output_dir / "frontal_ablation_results.csv", index=False)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Period: {df_FA['Start'].min().date()} to {df_FA['End'].max().date()}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Flowlines: {len(df_flow)} records")
    print(f"  - Fluxgates: {len(df_flux)} records")
    print(f"  - Intervals: {len(df_FA)} records")
    
    # Clear caches to free memory
    clear_all_caches()
    print("\nCaches cleared. Memory freed.")