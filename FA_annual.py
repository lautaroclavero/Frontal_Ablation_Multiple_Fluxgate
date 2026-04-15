
# -*- coding: utf-8 -*-
"""
Annual Frontal Ablation Calculation for Manso Glacier
======================================================

This module calculates annual frontal ablation rates following the
multiple-fluxgate method described in the paper.

The workflow consists of three steps:
    1. FA_flowlines: Calculate variables along individual flowlines
    2. FA_fluxgates: Combine adjacent flowlines into fluxgates
    3. FA_annual: Aggregate fluxgates to annual totals

Author: Lautaro Clavero
Email: lclavero@mendoza-conicet.gob.ar
"""

from math import sqrt
from pathlib import Path
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal
from typing import Union, List
from FA_annual_utils import (
    get_point_value,
    sample_raster_at_points,
    split_lines_by_id,
    nmad_error_from_stable_areas,
    compute_annual_thickness
)



def FA_flowlines(
    folder: Union[str, Path],
    path_multiflow: Union[str, Path],
    path_H_init: Union[str, Path],
    dhdt_map: dict,
    year_H_init: int,
    stable_shapes: List[gpd.GeoDataFrame],
    sliding_factor: float = 0.997,
    terminus_boundary: int = 25
) -> pd.DataFrame:
    """
    Calculate frontal ablation variables for each flowline and year.    
    Computes along each flowline:
        - Ut: Ice velocity at terminus (m/yr)
        - dL/dt: Terminus displacement rate (m/yr, positive = advance)
        - Uc: Frontal ablation rate = Ut - dL/dt (m/yr)
        - Ht: Ice thickness at terminus (m)
    
    Args:
        folder: Path to folder containing masks and velocity rasters
        path_multiflow: Path to flowline shapefile
        path_H_init: Path to reference thickness raster (2012)
        dhdt_map: Dictionary mapping years to dh/dt rates
        year_H_init: Reference year for thickness (e.g., 2012)
        stable_shapes: List of stable bedrock polygons for error estimation
        sliding_factor: relates surface speed with depth averaged speed (default: 0.997)
        terminus_boundary: Points behind terminus for averaging (default: 25-->75 m upglacier)
    
    Returns:
        DataFrame with flowline-level results
    """
    # Collect files 
    masks = []
    vels = []
    vels_final = []
    
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            fpath = str(Path(root, file_name).absolute())
            
            if 'mask' in file_name:
                masks.append(fpath)
            elif 'final' in file_name:  
                vels_final.append(fpath)
            else:
                vels.append(fpath)
    
    vels.sort()
    vels_final.sort()
    masks.sort()
    

    mask_ini = masks[:]      
    mask_fin = masks[1:]     
    
    # Pair velocities with masks
    paired_data = list(zip(vels, vels_final, mask_ini, mask_fin))
    
    flowlines = split_lines_by_id(path_multiflow)

    dfs = []
    
    for vel_raw, vel_final, mask_i, mask_f in tqdm(paired_data, desc="Processing years"):
        
        # Extract year from filename 
        # Assumes filename like "2013.tif" or "2013_final.tif"
        year = int(Path(vel_raw).stem[-4:])
        
        # Calculate velocity error once per year 
        err_Ut = nmad_error_from_stable_areas(vel_raw, stable_shapes)
        
        # Get mask geotransform for pixel size 
        ds = gdal.Open(mask_f)
        gt = ds.GetGeoTransform()
        pixelSizeX = gt[1]
        pixelSizeY = -gt[5]     
       
        for flowline in flowlines:
            line_A=flowline.copy()       #to avoid overwriting bugs
            line_B=flowline.copy()
            
            ID = int(line_A['LINE_ID'].mean())
            
            # Sample velocities and masks
            vel = get_point_value(line_A, vel_final)
            mask_start = sample_raster_at_points(line_A, mask_i)
            mask_end = sample_raster_at_points(line_B, mask_f)
            
            # Front detection (original logic)
            if len(mask_start.loc[mask_start.value != 1]) == 0 or \
               len(mask_end.loc[mask_end.value != 1]) == 0:
                continue
            
            # Find front positions (indices along flowline)
            a = mask_start.loc[mask_start.value != 1].index.values[0] - 1
            b = mask_end.loc[mask_end.value != 1].index.values[0] - 1
            c = a - terminus_boundary
            d = b - terminus_boundary
            
            mask_start_indices = list(mask_start.index.values)
            mask_end_indices = list(mask_end.index.values)
            
            # Find positions of c, a in mask_start_indices
            # Need to handle if c or d are not in the list (clamp to nearest)
            def find_closest_index(lst, val):
                if val in lst:
                    return lst.index(val)
                # Find nearest available index
                closest = min(lst, key=lambda x: abs(x - val))
                return lst.index(closest)
            
            i1 = find_closest_index(mask_start_indices, c)
            i2 = find_closest_index(mask_start_indices, a)
            i3 = find_closest_index(mask_end_indices, d)
            i4 = find_closest_index(mask_end_indices, b)
            
            # Front coordinates
            x_front_start, y_front_start = vel.iloc[i2].X, vel.iloc[i2].Y
            x_front_end, y_front_end = vel.iloc[i4].X, vel.iloc[i4].Y
            
            front_xy_init = (x_front_start, y_front_start)
            front_xy_final = (x_front_end, y_front_end)
            
            # Reference thickness
            H_start = sample_raster_at_points(line_A, path_H_init)
            
            # Calculate dL/dt 
            if a < b:  # Glacier advance
                rows_vel = vel.iloc[i1:i2]  
                idx_start, idx_end = i1, i2
                dL_dt = vel.iloc[i4].DIST - vel.iloc[i2].DIST
            else:  # Glacier retreat
                rows_vel = vel.iloc[i3:i4] 
                idx_start, idx_end = i3, i4
                dL_dt = vel.iloc[i4].DIST - vel.iloc[i2].DIST
            
            # Validate data completeness
            if len(rows_vel) == 0:
                continue
            if np.all(np.isnan(rows_vel.velocity)):
                continue
            
            #  thickness 
            rows_h, err_Ht = compute_annual_thickness(
                year, year_H_init, H_start, dhdt_map, idx_start, idx_end
            )
            
            # Calculate variables
            Ut = np.nanmedian(rows_vel.velocity) * sliding_factor
            Uc = Ut - dL_dt
            err_dL_dt = sqrt(pixelSizeX**2 + pixelSizeY**2)  
            err_Uc = sqrt(err_Ut**2 + err_dL_dt**2)
            Ht = rows_h.mean()
            
            dfs.append({
                'Year': year,
                'ID': ID,
                'Ut': Ut,
                'err_Ut': err_Ut,
                'dL/dt': dL_dt,
                'err_dL/dt': err_dL_dt,
                'Uc': Uc,
                'err_Uc': err_Uc,
                'Ht': Ht,
                'err_Ht': err_Ht,
                'XY_front_start': front_xy_init,
                'XY_front_end': front_xy_final,
                'px_size_mask': pixelSizeX
            })
    
    return pd.DataFrame(dfs)


def FA_fluxgates(flowline_df: pd.DataFrame, rho: float = 0.5) -> pd.DataFrame:
    """
    Calculate frontal ablation for each fluxgate. 
    A fluxgate is defined by two adjacent flowlines. For each fluxgate, calculates:
        - Uc: Mean frontal ablation rate (m/yr)
        - Hc: Mean ice thickness (m)
        - Wi: Fluxgate width (m)
        - Qc: Frontal ablation volume flux = Uc * Hc * Wi (m³/yr)
    
    Args:
        flowline_df: DataFrame from FA_flowlines()
        rho: Spatial correlation coefficient between neighboring flowlines.
             Default 0.5 as used in the paper (conservative estimate).    
    Returns:
        DataFrame with fluxgate-level results
    """
    # Correlation factor for error propagation 
    correlation_factor = sqrt((1 + rho) / 2)
    
    results = []
    
    for year, df_year in flowline_df.groupby('Year'):
        df_year = df_year.sort_values('ID').reset_index(drop=True)
        n = len(df_year)
        
        if n < 2:
            print(f"Year {year}: less than 2 flowlines, skipping fluxgates")
            continue
        
        for i in range(1, n):
            j = i - 1
            
            # Get coordinates 
            xi_s, yi_s = df_year.iloc[i].XY_front_start
            xj_s, yj_s = df_year.iloc[j].XY_front_start
            xi_e, yi_e = df_year.iloc[i].XY_front_end
            xj_e, yj_e = df_year.iloc[j].XY_front_end
            
            # Widths wa and wb
            wa = sqrt((xi_s - xj_s)**2 + (yi_s - yj_s)**2)
            wb = sqrt((xi_e - xj_e)**2 + (yi_e - yj_e)**2)
            
            # Fluxgate width
            w = (wa + wb) / 2
            
            # Width uncertainty
            sigma_geom = abs(wa - wb) / 2
            mask_res = df_year.iloc[j].px_size_mask
            sigma_pos = mask_res * 0.5
            err_w = sqrt(sigma_geom**2 + sigma_pos**2)
            
            # Frontal ablation rate 
            Uc = (df_year.iloc[i].Uc + df_year.iloc[j].Uc) / 2
            err_Uc = df_year.iloc[i].err_Uc * correlation_factor
            
            # Thickness 
            Hc = (df_year.iloc[i].Ht + df_year.iloc[j].Ht) / 2
            err_Hc = df_year.iloc[i].err_Ht * correlation_factor
            
            # Area (original: Si = Hc * w)
            area = Hc * w
            err_area = sqrt((err_Hc**2 * area**2 / Hc**2) + (err_w**2 * area**2 / w**2))
            
            # Volume flux (original: Qc = Uc * Hc * w)
            Qc = Uc * Hc * w
            err_Qc = Qc * sqrt((err_Uc**2 / Uc**2) +
                               (err_Hc**2 / Hc**2) +
                               (err_w**2 / w**2))
            
            # Gate ID 
            gate_id = int(f"{int(df_year.iloc[j].ID):02d}{int(df_year.iloc[i].ID):02d}")
            
            results.append({
                'Year': year,
                'Gate_ID': gate_id,
                'Uc': Uc,
                'err_Uc': err_Uc,
                'Hc': Hc,
                'err_Hc': err_Hc,
                'A_m2': area,
                'err_A': err_area,
                'Wa': wa,
                'Wb': wb,
                'Wi': w,
                'err_Wi': err_w,
                'Qc': Qc,
                'err_Qc': err_Qc
            })
    
    return pd.DataFrame(results)


def FA_annual(fluxgate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fluxgates to annual totals.    
    Sums Qc across all fluxgates for each year to get total
    frontal ablation volume.    
    Args:
        fluxgate_df: DataFrame from FA_fluxgates()    
    Returns:
        DataFrame with annual frontal ablation totals (Mm³/year)
    """
    results = []
    
    for year, df_year in fluxgate_df.groupby('Year'):
        Qc_total = df_year['Qc'].sum()
        err_total = sqrt((df_year['err_Qc']**2).sum())
        
        results.append({
            'Year': year,
            'Frontal_ablation_Mm3': Qc_total / 1e6,
            'Err_Frontal_ablation_Mm3': err_total / 1e6
        })
    
    return pd.DataFrame(results).sort_values('Year')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("Annual Frontal Ablation Calculation")
    print("=" * 60)
    
    # ==================== INPUT PATHS ====================
    
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "Input_Data_Manso"
    
    # Stable bedrock polygons 
    stable_shapes = []
    stable_zones_dir = DATA_DIR / "geometry" / "stable_zones"
    for shp_file in sorted(stable_zones_dir.glob("stable_*.shp")):
        try:
            stable_shapes.append(gpd.read_file(shp_file))
            print(f"  Loaded: {shp_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {shp_file.name}: {e}")
    
    if len(stable_shapes) == 0:
        raise ValueError(f"No stable zone shapes found in {stable_zones_dir}")
    
    # Data folders
    path_landsat = DATA_DIR / "data_Manso_annual_Landsat"
    path_sentinel = DATA_DIR / "data_Manso_annual_Sentinel"
    
    # Reference thickness (Zorzut et al., 2020)
    H2012 = DATA_DIR / "thickness" / "thickness_2012_Manso.tif"
    
    # Flowlines shapefile
    flowlines_shp = DATA_DIR / "geometry" / "Flowlines.shp"
    
    # Annual dh/dt rates (m/year) from Pleiades DEM differencing
    dhdt_map = {
        (2013, 2016): {"dhdt": -6.78, "err": 0.21},
        2017: {"dhdt": -9.88, "err": 0.28},
        2018: {"dhdt": -9.64, "err": 0.37},
        2019: {"dhdt": -5.95, "err": 0.34},
        2020: {"dhdt": -5.52, "err": 0.45},
        2021: {"dhdt": -8.34, "err": 0.63},
        2022: {"dhdt": -6.79, "err": 0.57},
        2023: {"dhdt": -7.31, "err": 0.36}
    }
    
    # Check if input directories exist
    if not path_landsat.exists():
        print(f"\nERROR: Landsat data folder not found: {path_landsat}")
        print("Please ensure the data folder structure is correct.")
        exit(1)
    
    if not path_sentinel.exists():
        print(f"\nERROR: Sentinel data folder not found: {path_sentinel}")
        print("Please ensure the data folder structure is correct.")
        exit(1)
    
    if not H2012.exists():
        print(f"\nERROR: Thickness raster not found: {H2012}")
        exit(1)
    
    if not flowlines_shp.exists():
        print(f"\nERROR: Flowlines shapefile not found: {flowlines_shp}")
        exit(1)
    
    # ==================== RUN CALCULATION ====================
    
    # Process Landsat data
    print("\n1. Processing Landsat data...")
    flow_landsat = FA_flowlines(
        path_landsat, flowlines_shp, H2012,
        dhdt_map, 2012, stable_shapes
    )
    print(f"   Landsat records: {len(flow_landsat)}")
    
    # Process Sentinel data
    print("\n2. Processing Sentinel data...")
    flow_sentinel = FA_flowlines(
        path_sentinel, flowlines_shp, H2012,
        dhdt_map, 2012, stable_shapes
    )
    print(f"   Sentinel records: {len(flow_sentinel)}")
    
    # Combine and filter 
    print("\n3. Combining and filtering results...")
    df_flow = pd.concat([flow_landsat, flow_sentinel], ignore_index=True)
    
    n_before = len(df_flow)
    # Original filter: (Ut >= 0) & (Uc >= 0) & (dL/dt > -200)
    df_flow = df_flow[(df_flow['Ut'] >= 0) & 
                      (df_flow['Uc'] >= 0) & 
                      (df_flow['dL/dt'] > -200)]
    print(f"   Removed {n_before - len(df_flow)} invalid records")
    print(f"   Remaining: {len(df_flow)} flowline records")
    
    # Calculate fluxgates
    print("\n4. Calculating fluxgates...")
    df_fluxgate = FA_fluxgates(df_flow, rho=0.5)
    print(f"   Fluxgates: {len(df_fluxgate)}")
    
    # Calculate annual totals
    print("\n5. Calculating annual totals...")
    df_annual = FA_annual(df_fluxgate)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS - Annual Frontal Ablation (Mm³/year)")
    print("=" * 60)
    print(df_annual.to_string(index=False))
    
    # ==================== SAVE RESULTS ====================
    
    # Create output directory
    output_dir = SCRIPT_DIR / "output" / "annual"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataframes
    df_flow.to_csv(output_dir / "annual_flowlines_results.csv", index=False)
    df_fluxgate.to_csv(output_dir / "annual_fluxgates_results.csv", index=False)
    df_annual.to_csv(output_dir / "annual_frontal_ablation.csv", index=False)
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Flowlines: {len(df_flow)} records")
    print(f"  - Fluxgates: {len(df_fluxgate)} records")
    print(f"  - Annual totals: {len(df_annual)} years")