
# -*- coding: utf-8 -*-

"""
Auxiliar functions for FA subseasonal estimation
@author: Lautaro Clavero 
"""

from pathlib import Path
import os
from datetime import datetime
import numpy as np
import pandas as pd 
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
from osgeo import gdal
import scipy.interpolate as interp
import rasterio
from dateutil.relativedelta import relativedelta
from functools import lru_cache


_RASTER_CACHE = {}          # dataset -> gdal dataset
_RASTER_ARRAY_CACHE = {}    # path -> numpy array
_GT_CACHE = {}              # path -> geotransform
_THICKNESS_CACHE = {}       # key -> (H, err_H) 
_NMAD_CACHE = {}            # key -> nmad value
_VEL_CACHE = {}             # key -> velocity array


def clear_all_caches():
    """Clear all global caches to free memory."""
    global _RASTER_CACHE, _RASTER_ARRAY_CACHE, _GT_CACHE, _THICKNESS_CACHE, _NMAD_CACHE, _VEL_CACHE
    _RASTER_CACHE.clear()
    _RASTER_ARRAY_CACHE.clear()
    _GT_CACHE.clear()
    _THICKNESS_CACHE.clear()
    _NMAD_CACHE.clear()
    _VEL_CACHE.clear()


def get_raster_cached(raster_path):
    """Return dataset of raster with cache."""
    if raster_path not in _RASTER_CACHE:
        _RASTER_CACHE[raster_path] = gdal.Open(raster_path)
    return _RASTER_CACHE[raster_path]


def get_raster_array_cached(raster_path):
    """Return array of raster with cache."""
    if raster_path not in _RASTER_ARRAY_CACHE:
        ds = get_raster_cached(raster_path)
        _RASTER_ARRAY_CACHE[raster_path] = ds.ReadAsArray().astype(np.float32)
    return _RASTER_ARRAY_CACHE[raster_path]


def get_geotransform_cached(raster_path):
    """Return geotransform of raster with cache."""
    if raster_path not in _GT_CACHE:
        ds = get_raster_cached(raster_path)
        _GT_CACHE[raster_path] = ds.GetGeoTransform()
    return _GT_CACHE[raster_path]


def coefficients(date_ini_str, date_fin_str, velocities):
    """Calculate weighting coefficients for monthly velocity grids."""
    date_ini = datetime.strptime(date_ini_str, '%Y%m%d')
    date_fin = datetime.strptime(date_fin_str, '%Y%m%d')

    months = set()
    for file_name in velocities:
        base_name = os.path.basename(file_name)
        if 'final' in base_name:
            month = base_name[:6]
        else:
            month = file_name[-10:-4]
        if len(month) != 6:
            print("Error: Unexpected month in file:", file_name)
            continue
        months.add(month)

    months = sorted(months)
    days_per_month = {}

    for month in months:
        month_obj = datetime.strptime(month, '%Y%m')
        first_day = max(date_ini, month_obj.replace(day=1))
        last_day = min(date_fin, month_obj.replace(day=1) + relativedelta(months=1, days=-1))
        days_per_month[month] = (last_day - first_day).days + 1

    total_days = sum(days_per_month.values())
    return [days_per_month[month] / total_days for month in months]


def get_velocity(vel_paths: list, coefs: list) -> np.ndarray:
    """Return velocity array (weighted average if multiple grids).    
    Args:
        vel_paths: List of raster file paths
        coefs: Weighting coefficients for each raster    
    Returns:
        2D numpy array of weighted velocities
    """
    if not vel_paths:
        raise ValueError("Velocity paths list is empty")
    
    # Create unique key for cache
    cache_key = (tuple(vel_paths), tuple(coefs))
    
    if cache_key in _VEL_CACHE:
        return _VEL_CACHE[cache_key]
    
    if len(vel_paths) == 1:
        result = get_raster_array_cached(vel_paths[0])
    else:
        # Weighted average
        weighted = np.zeros_like(get_raster_array_cached(vel_paths[0]))
        for path, coef in zip(vel_paths, coefs):
            weighted += coef * get_raster_array_cached(path)
        result = weighted
    
    _VEL_CACHE[cache_key] = result
    return result


def get_point_value(df_point, array, raster_path, interp_method='nearest', nan_value=-9999):
    """Extract point values from raster with interpolation."""
    geot = get_geotransform_cached(raster_path)
    raster = get_raster_cached(raster_path)
    Xsize, Ysize = raster.RasterXSize, raster.RasterYSize

    data = array.copy()
    data[data == nan_value] = np.nan

    extent = [geot[0], geot[0] + np.round(geot[1], 3) * Xsize,
              geot[3], geot[3] + np.round(geot[5], 3) * Ysize]

    Xs = np.linspace(extent[0] + np.round(geot[1], 3), extent[1], Xsize)
    Ys = np.linspace(extent[2] + np.round(geot[5], 3), extent[3], Ysize)
    XX, YY = np.meshgrid(Xs, Ys)

    XY = np.vstack((XX.flatten(), YY.flatten())).T
    Z = data.flatten()

    df_point['velocity'] = interp.griddata(XY, Z, (df_point.X, df_point.Y), method=interp_method)
    return df_point


def split_lines_by_id(multiline_path):
    """Split MultiPoint shapefile into individual lines by LINE_ID."""
    multiline = gpd.read_file(multiline_path)
    return [multiline[multiline['LINE_ID'] == lid] for lid in sorted(multiline['LINE_ID'].unique())]


def sample_raster_at_points(pts: gpd.GeoDataFrame, raster_path: str) -> pd.DataFrame:
    """Extract raster values at point locations without interpolation. 
    Args:
        pts: GeoDataFrame with points containing columns 'LINE_ID', 'ID', 'DIST', 'X', 'Y'
        raster_path: Path to raster file (GeoTIFF)    
    Returns:
        DataFrame with original columns plus 'value' column with raster values
    """
    pts_copy = pts[['LINE_ID', 'ID', 'DIST', 'X', 'Y']].copy()
    pts_copy.index = range(len(pts_copy))
    coords = [(x, y) for x, y in zip(pts_copy.X, pts_copy.Y)]

    with rasterio.open(raster_path) as src:
        pts_copy['value'] = [x[0] for x in src.sample(coords)]
    
    return pts_copy


def nmad_error(raster_path, raster_array, shape_paths):
    """Calculate mean NMAD across stable bedrock polygons. """
    # Create unique key for cache
    cache_key = (raster_path, tuple(shape_paths))
    
    if cache_key in _NMAD_CACHE:
        return _NMAD_CACHE[cache_key]
    
    dataset = rioxarray.open_rasterio(raster_path)
    nmad_values = []

    for shape_path in shape_paths:
        shape = gpd.read_file(shape_path)
        shape_geom = [mapping(geom) for geom in shape.geometry]
        clipped = dataset.rio.clip(shape_geom, shape.crs, all_touched=True)
        values = clipped.data[~np.isnan(clipped.data)]
        
        if len(values) > 0:
            median = np.median(values)
            nmad = 1.4826 * np.median(np.abs(values - median))
            nmad_values.append(nmad)

    result = np.mean(nmad_values) if nmad_values else np.nan
    _NMAD_CACHE[cache_key] = result
    return result

# --------------------------------------------------------------------
#              INTERVAL GENERATION FUNCTIONS
# --------------------------------------------------------------------

def _collect_files(folder):
    """
    Extract masks and velocities from folder.
    Returns common data structures used by both version A and B.
    """
    mask_paths, mask_dates, mask_datetimes = [], [], []
    raw_vels, final_vels = [], []
    final_vel_months, months = [], []
    
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            fpath = str(Path(root, file_name).absolute())
            
            if 'mask' in file_name:
                mask_paths.append(fpath)
                date_mask = file_name[-17:-9]
                mask_dates.append(date_mask)
                mask_datetimes.append(datetime.strptime(date_mask, '%Y%m%d'))
                
            elif 'final' in file_name:
                base_name = os.path.basename(fpath)  # '201706_final.tif'
                month = base_name[:6]  # '201706'
                months.append(month)
                final_vel_months.append(int(month))
                final_vels.append((month, fpath))
                
            else:  # raw velocities
                month = int(file_name[-12:-4])
                raw_vels.append((month, fpath))
    
    # Filter raw velocities that correspond to final velocities
    raw_vels_filtered = [path for month, path in raw_vels if month in final_vel_months]
    masks = list(zip(mask_paths, mask_dates, mask_datetimes))
    
    return masks, months, final_vels, raw_vels_filtered


def _build_interval(mask_ini_info, mask_fin_info, months_in_interval, final_vels, raw_vels_filtered):
    """
    Build interval dictionary from mask info and filtered months.
    """
    filtered_final_vels = [fpath for month, fpath in final_vels if month in months_in_interval]
    months_set = set(months_in_interval)
    filtered_raw_vels = [path for path in raw_vels_filtered 
                        if any(month in path for month in months_set)]
    
    coefs = coefficients(mask_ini_info[1], mask_fin_info[1], filtered_final_vels)
    
    return {
        'date_ini': mask_ini_info[1],
        'date_fin': mask_fin_info[1],
        'interval_days': (mask_fin_info[2] - mask_ini_info[2]).days + 1,
        'months': months_in_interval,
        'coefficients': coefs,
        'mask_ini': mask_ini_info[0],
        'mask_fin': mask_fin_info[0],
        'final_vels': filtered_final_vels,
        'raw_vels': filtered_raw_vels,
    }


def _get_months_in_interval(mask_ini_date, mask_fin_date, all_months):
    """Return sorted list of months between two mask dates."""
    return [m for m in all_months 
            if mask_ini_date.strftime('%Y%m') <= m <= mask_fin_date.strftime('%Y%m')]


def list_files_FA_A(folder, min_days, max_days):
    """
    Version A: No overlapping intervals.
    Advantage: No overlap.
    Disadvantage: Does not maximize number of intervals.
    """
    masks, all_months, final_vels, raw_vels_filtered = _collect_files(folder)
    interval_dicts = []
    
    for idx_ini, mask_ini_info in enumerate(masks):
        mask_ini_date = mask_ini_info[2]
        
        for idx_fin in range(idx_ini, len(masks)):
            mask_fin_date = masks[idx_fin][2]
            days_diff = (mask_fin_date - mask_ini_date).days
            
            if days_diff >= min_days and days_diff <= max_days:
                months_in_interval = _get_months_in_interval(mask_ini_date, mask_fin_date, all_months)
                
                if months_in_interval:
                    interval_dicts.append(_build_interval(
                        mask_ini_info, masks[idx_fin], months_in_interval, 
                        final_vels, raw_vels_filtered
                    ))
                break  # Take only the first valid interval for this start
    
    # Filter A: no overlapping intervals 
    filtered = []
    unique_dates = set()
    for interval in sorted(interval_dicts, key=lambda x: datetime.strptime(x['date_ini'], '%Y%m%d')):
        if all(interval['date_ini'] > date_fin for date_fin in unique_dates):
            filtered.append(interval)
            unique_dates.add(interval['date_fin'])
    
    return [d for d in filtered if d['months']]


def list_files_FA_B(folder, min_days, max_days):
    """
    Version B: Partial overlap allowed.
    """
    masks, all_months, final_vels, raw_vels_filtered = _collect_files(folder)
    interval_dicts = []
    
    for idx_ini, mask_ini_info in enumerate(masks):
        mask_ini_date = mask_ini_info[2]
        
        for idx_fin in range(idx_ini, len(masks)):
            mask_fin_date = masks[idx_fin][2]
            days_diff = (mask_fin_date - mask_ini_date).days
            
            if days_diff >= min_days and days_diff <= max_days:
                months_in_interval = _get_months_in_interval(mask_ini_date, mask_fin_date, all_months)
                
                if months_in_interval:
                    interval_dict = _build_interval(
                        mask_ini_info, masks[idx_fin], months_in_interval,
                        final_vels, raw_vels_filtered
                    )
                    
                    interval_contained = False
                    for existing in interval_dicts[:]:  
                        if set(interval_dict['months']).issubset(set(existing['months'])):
                            interval_contained = True
                            break
                        elif set(existing['months']).issubset(set(interval_dict['months'])):
                            interval_dicts.remove(existing)
                            break  
                    
                    if not interval_contained:
                        interval_dicts.append(interval_dict)
                break
    
    return [d for d in interval_dicts if d['months']]


def list_files_FA(folder, min_days, max_days):
    """
    Combines intervals from version A and version B.
    Returns unique intervals (same as original list_calving).
    """
    intervals_A = list_files_FA_A(folder, min_days, max_days)
    intervals_B = list_files_FA_B(folder, min_days, max_days)
    
    combined = intervals_A + intervals_B
    
    # Remove duplicates
    unique_intervals = []
    seen = set()
    for interval in combined:
        key = (interval['date_ini'], interval['date_fin'])
        if key not in seen:
            seen.add(key)
            unique_intervals.append(interval)
    
    return unique_intervals


# ============================================================================
# THICKNESS UPDATE FUNCTIONS 
# ============================================================================

def find_dhdt_for_date(date, dhdt_map):
    """
    Return dhdt, error, period_start, period_end for given date.
    EXACT copy of original buscar_dhdt_para_fecha.
    """
    for (start, end), data in dhdt_map.items():
        if start <= date < end:
            return data["dhdt"], data["err"], start, end
    print(f" No dh/dt found for date {date}")
    return None, None, None, None


def update_thickness(H_initial, dhdt, err_dhdt, days_interval):
    """
    Advance thickness applying dhdt for given days.
    EXACT copy of original actualizar_espesor_intervalo.
    """
    delta_years = days_interval / 365.25
    H_final = H_initial + dhdt * delta_years
    err_H = err_dhdt * delta_years
    return H_final, err_H


def calculate_subseasonal_thickness(target_date, zero_date_str, H_zero_data, dhdt_map, idx_start, idx_end):
    """
    Calculate thickness by advancing period by period.
    EXACT copy of original calcular_espesor_subestacional.
    """
    # Extract initial thickness
    H = H_zero_data.value.iloc[idx_start:idx_end].values.astype(float)
    err_H_total = np.full_like(H, 31.0)  # RMSE from Zorzut et al. 2020
    
    # Convert zero date string to datetime
    zero_date = datetime(int(zero_date_str[:4]), int(zero_date_str[4:6]), int(zero_date_str[6:8]))
    current_date = zero_date
    
    # If target is exactly the zero date
    if target_date == zero_date:
        return H, err_H_total
    
    # Advance period by period 
    while current_date < target_date:
        dhdt, err_dhdt, ini, fin = find_dhdt_for_date(current_date, dhdt_map)
        
        if dhdt is None:
            # Skip if no dh/dt found 
            current_date = min(fin if fin else target_date, target_date)
            continue
        
        # End of this sub-interval
        date_limit = min(fin, target_date)
        days = (date_limit - current_date).days
        
        # Advance thickness
        H, err = update_thickness(H, dhdt, err_dhdt, days)
        err_H_total = np.sqrt(err_H_total**2 + err**2)
        
        # Advance date
        current_date = date_limit
    
    return H, err_H_total