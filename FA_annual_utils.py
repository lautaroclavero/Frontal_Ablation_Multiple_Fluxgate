
# -*- coding: utf-8 -*-
"""
Auxiliary functions for annual frontal ablation (FA) calculations.
Author: Lautaro Clavero
Email: lclavero@mendoza-conicet.gob.ar
"""

import warnings
from math import sqrt
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import xarray
import scipy.interpolate as interp
from osgeo import gdal
warnings.filterwarnings("ignore", category=UserWarning, module=__name__)

# ============================================================================
# Constants
# ============================================================================

# RMSE for Manso Glacier thickness in 2012 (Zorzut et al., 2020)
INITIAL_THICKNESS_RMSE = 31.0  # meters

# UTM projection for Northern Patagonia
UTM_19S_EPSG = "EPSG:32719"

# No-data value for velocity rasters
NODATA_VALUE = -9999


# ============================================================================
# Vector utilities
# ============================================================================

def split_lines_by_id(path_multiline: Union[str, Path]) -> List[gpd.GeoDataFrame]:
    """
    Split a MultiLineString shapefile into individual lines by LINE_ID.
    
    Args:
        path_multiline: Path to the shapefile containing flowlines
        
    Returns:
        List of GeoDataFrames, each containing one flowline
    """
    multiline = gpd.read_file(path_multiline)
    line_ids = sorted(multiline["LINE_ID"].unique())
    return [multiline[multiline["LINE_ID"] == lid] for lid in line_ids]


# ============================================================================
# Raster sampling
# ============================================================================

def sample_raster_at_points(pts: pd.DataFrame, raster_path: Union[str, Path]) -> pd.DataFrame:
    """
    Extract raster values at point locations WITHOUT interpolation.    
    Equivalent to QGIS Point Sampling Tool.    
    Args:
        pts: DataFrame with columns ['LINE_ID', 'ID', 'DIST', 'X', 'Y']
        raster_path: Path to raster file.         
    Returns:
        DataFrame with original columns plus 'value' column with raster values
    """
    pts = pts[['LINE_ID', 'ID', 'DIST', 'X', 'Y']].copy()
    pts.index = range(len(pts))
    
    coords = [(x, y) for x, y in zip(pts.X, pts.Y)]
    
    with rasterio.open(raster_path) as src:
        pts["value"] = [val[0] for val in src.sample(coords)]
    
    return pts


def get_point_value(df_point: pd.DataFrame, raster_path: Union[str, Path],
                    raster_band: int = 1, interp_method: str = 'nearest') -> pd.DataFrame:
    """
    Interpolate raster values at point locations using scipy griddata.    
    This function performs bilinear or nearest-neighbor interpolation,
    which is useful when points don't align exactly with pixel centers.    
    Args:
        df_point: DataFrame with 'X' and 'Y' columns
        raster_path: Path to raster file (GeoTIFF)
        raster_band: Band number to read (default: 1)
        interp_method: Interpolation method ('nearest', 'linear', 'cubic')        
    Returns:
        DataFrame with added 'velocity' column containing interpolated values
    """
    ds = gdal.Open(str(raster_path))
    geot = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    
    # Read raster data
    data = ds.GetRasterBand(raster_band).ReadAsArray(0, 0, xsize, ysize)
    data = data.astype(np.float32)
    data[data == NODATA_VALUE] = np.nan
    
    # Create coordinate grids
    extent = [
        geot[0],
        geot[0] + np.round(geot[1], 3) * xsize,
        geot[3],
        geot[3] + np.round(geot[5], 3) * ysize
    ]
    
    xs = np.linspace(extent[0] + np.round(geot[1], 3), extent[1], xsize)
    ys = np.linspace(extent[2] + np.round(geot[5], 3), extent[3], ysize)
    XX, YY = np.meshgrid(xs, ys)
    
    # Interpolate
    XY = np.vstack((XX.flatten(), YY.flatten())).T
    Z = data.flatten()
    
    df_point["velocity"] = interp.griddata(
        XY, Z, (df_point.X, df_point.Y), method=interp_method
    )
    
    return df_point


# ============================================================================
# Error estimation
# ============================================================================

def extract_raster_values_over_polygon(raster_path: Union[str, Path],
                                       polygon: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract all raster values within a polygon.    
    Args:
        raster_path: Path to raster file
        polygon: GeoDataFrame containing a single polygon        
    Returns:
        1D array of raster values inside the polygon
    """
    dataset = rioxarray.open_rasterio(raster_path)
    
    # Set CRS if missing (UTM 19S for Northern Patagonia)
    if not dataset.rio.crs:
        dataset.rio.write_crs(UTM_19S_EPSG, inplace=True)
    
    clipped = dataset.rio.clip(
        polygon.geometry.values,
        polygon.crs,
        all_touched=True
    )
    
    return np.ravel(xarray.DataArray.to_numpy(clipped))


def nmad_error_from_stable_areas(raster_path: Union[str, Path],
                                  stable_shapes: List[gpd.GeoDataFrame]) -> float:
    """
    Calculate velocity error as NMAD over stable bedrock areas.    
    NMAD (Normalized Median Absolute Deviation) is a robust estimator
    of scatter, less sensitive to outliers than standard deviation.    
    Args:
        raster_path: Path to velocity raster
        stable_shapes: List of GeoDataFrames with stable bedrock polygons        
    Returns:
        NMAD error value (same units as input raster)
    """
    def nmad(data: np.ndarray) -> float:
        """Calculate Normalized Median Absolute Deviation."""
        median = np.nanmedian(data)
        return 1.4826 * np.nanmedian(np.abs(data - median))
    
    # Extract values from all stable areas
    all_values = np.concatenate([
        extract_raster_values_over_polygon(raster_path, shape)
        for shape in stable_shapes
    ])
    
    return nmad(all_values)


# ============================================================================
# Ice thickness evolution
# ============================================================================

def get_dhdt_for_year(year: int, dhdt_map: dict) -> Tuple[float, float]:
    """
    Get dh/dt rate and error for a specific year.    
    Args:
        year: Calendar year (e.g., 2015)
        dhdt_map: Dictionary mapping years or year ranges to dh/dt values.
                  Format: {year: {"dhdt": value, "err": error}} or
                          {(start, end): {"dhdt": value, "err": error}}                  
    Returns:
        Tuple of (dhdt_rate, dhdt_error). Returns (None, None) if not found.
    """
    for key, val in dhdt_map.items():
        if isinstance(key, tuple):
            start, end = key
            if start <= year <= end:
                return val["dhdt"], val["err"]
        else:
            if year == key:
                return val["dhdt"], val["err"]
    return None, None


def compute_annual_thickness(
    current_year: int,
    initial_year: int,
    initial_thickness_data: pd.DataFrame,
    dhdt_map: dict,
    idx_start: int,
    idx_end: int,
    initial_thickness_error: float = INITIAL_THICKNESS_RMSE
) -> Tuple[pd.Series, float]:
    """
    Update ice thickness from initial_year to current_year applying dh/dt year by year.    
    Propagates thickness forward in time using annual dh/dt rates. The error
    is propagated in quadrature.    
    IMPORTANT: 
        - For year range, applies dh/dt from initial_year to current_year-1
        - Uses the dh/dt value associated with the START year of each interval    
    Args:
        current_year: Target year (e.g., 2020)
        initial_year: Reference year with known thickness (e.g., 2012)
        initial_thickness_data: DataFrame with thickness values at initial_year
                                Must have 'value' column
        dhdt_map: Dictionary mapping years to {"dhdt": value, "err": error}
        idx_start: Starting index for thickness extraction
        idx_end: Ending index for thickness extraction
        initial_thickness_error: RMSE of initial thickness (meters)        
    Returns:
        Tuple of (thickness_series, thickness_error)
    """
    # Extract thickness values for the specified indices (EXACT same logic as original)
    block = initial_thickness_data.iloc[idx_start:idx_end]
    
    if hasattr(block, "columns") and ("value" in block.columns):
        hi = block["value"].astype(float).to_numpy()
        idx = block.index
    else:
        hi = block.squeeze().astype(float).to_numpy()
        idx = block.index if hasattr(block, "index") else None
    
    # Initialize thickness and variance
    H = hi.copy()
    var_H = initial_thickness_error ** 2
    
    # # Apply dh/dt year by year 
    if current_year < initial_year:
        warnings.warn(
            f"Current year {current_year} < initial year {initial_year}. "
            "No thickness update performed."
        )
    else:
        # Original loops from initial_year to current_year - 1
        for y in range(initial_year, current_year):
            dhdt_y, err_y = get_dhdt_for_year(y, dhdt_map)
            
            if dhdt_y is None:
                warnings.warn(f"No dh/dt assigned for year {y}. Skipping.")
                continue
            
            H += dhdt_y
            var_H += err_y ** 2
   
    thickness_error = sqrt(var_H)

    try:
        import pandas as pd
        thickness_series = pd.Series(H, index=idx)
    except Exception:
        thickness_series = H
    
    return thickness_series, thickness_error