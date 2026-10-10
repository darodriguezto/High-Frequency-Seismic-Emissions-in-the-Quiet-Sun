#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:05:19 2026

@author: daniel
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt
)

from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops

def load_egression_map(filepath):
    """
    Load a single egression power map from a FITS file.
    """
    filepath = Path(filepath)

    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(float).copy()
        header = hdul[0].header.copy()

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D map, got shape {data.shape}"
        )

    return data, header


def preprocess_egression_map(data, exclude_zeros=True):
    """
    Replace invalid pixels with NaN without changing
    the original map geometry.
    """
    clean_data = np.asarray(data, dtype=float).copy()

    valid_mask = np.isfinite(clean_data)

    if exclude_zeros:
        valid_mask &= clean_data != 0

    clean_data[~valid_mask] = np.nan

    return clean_data, valid_mask


def load_egression_maps(base_path, files, exclude_zeros=True):
    """
    Load and preprocess multiple egression power maps.

    Parameters
    ----------
    base_path : str or Path
        Directory containing the FITS files.
    files : dict
        Dictionary mapping frequency bands to FITS filenames.
    exclude_zeros : bool
        Whether zero-valued pixels should be excluded.

    Returns
    -------
    egression_data : dict
        Dictionary containing the map, header, and valid
        pixel mask for each frequency band.
    """
    base_path = Path(base_path)
    egression_data = {}

    for frequency, filename in files.items():

        filepath = base_path / filename

        data, header = load_egression_map(filepath)

        clean_data, valid_mask = preprocess_egression_map(
            data,
            exclude_zeros=exclude_zeros
        )

        egression_data[frequency] = {
            "filename": filename,
            "data": clean_data,
            "header": header,
            "valid_mask": valid_mask
        }

    return egression_data
def gaussian_smooth_valid(data, valid_mask, sigma=2):
    """
    Apply normalized Gaussian smoothing over valid pixels only.

    Invalid pixels do not contribute to the convolution,
    and the original map geometry is preserved.

    Parameters
    ----------
    data : np.ndarray
        Egression power map containing NaNs at invalid pixels.
    valid_mask : np.ndarray
        Boolean mask identifying valid pixels.
    sigma : float
        Gaussian standard deviation in pixels.

    Returns
    -------
    smooth : np.ndarray
        Smoothed egression power map with invalid pixels
        preserved as NaN.
    """
    if data.shape != valid_mask.shape:
        raise ValueError(
            "Data and valid_mask must have the same shape."
        )

    # Replace invalid pixels with zero for convolution
    values = np.where(valid_mask, data, 0.0)

    # Gaussian convolution of the data
    smooth_values = gaussian_filter(
        values,
        sigma=sigma
    )

    # Gaussian convolution of the validity weights
    weights = gaussian_filter(
        valid_mask.astype(float),
        sigma=sigma
    )

    # Initialize the output with NaNs
    smooth = np.full(
        data.shape,
        np.nan,
        dtype=float
    )

    # Normalize only where valid information is available
    good = valid_mask & (weights > 0)

    smooth[good] = (
        smooth_values[good] / weights[good]
    )

    return smooth
def detect_egression_bubbles(
    smooth,
    valid_mask,
    percentile_threshold=80,
    closing_iterations=2,
    min_size=500,
    min_distance=30
):
    """
    Detect and characterize bright structures in a smoothed
    egression power map using thresholding and watershed.

    Parameters
    ----------
    smooth : np.ndarray
        Gaussian-smoothed egression power map.
    valid_mask : np.ndarray
        Boolean mask of valid pixels.
    percentile_threshold : float
        Percentile used to identify bright pixels.
    closing_iterations : int
        Number of binary closing iterations.
    min_size : int
        Minimum connected-component size in pixels.
    min_distance : int
        Minimum distance between watershed seeds in pixels.

    Returns
    -------
    results : dict
        Intermediate masks, segmentation labels,
        and detected bubble properties.
    """

    if smooth.shape != valid_mask.shape:
        raise ValueError(
            "Smooth map and valid_mask must have the same shape."
        )

    valid = valid_mask & np.isfinite(smooth)

    if not np.any(valid):
        raise ValueError("No valid pixels available for detection.")

    # ========================================================
    # 1. Threshold bright regions
    # ========================================================

    threshold = np.percentile(
        smooth[valid],
        percentile_threshold
    )

    bright_mask = (
        (smooth > threshold)
        & valid
    )

    # ========================================================
    # 2. Morphological cleaning
    # ========================================================

    clean_mask = binary_closing(
        bright_mask,
        iterations=closing_iterations
    )

    clean_mask = binary_fill_holes(clean_mask)

    clean_mask = remove_small_objects(
        clean_mask,
        min_size=min_size
    )

    clean_mask &= valid

    # ========================================================
    # 3. Distance transform
    # ========================================================

    distance = distance_transform_edt(clean_mask)

    # ========================================================
    # 4. Detect candidate centers
    # ========================================================

    coords = peak_local_max(
        distance,
        labels=clean_mask,
        min_distance=min_distance
    )

    markers = np.zeros_like(distance, dtype=int)

    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    # ========================================================
    # 5. Watershed segmentation
    # ========================================================

    labels = watershed(
        -distance,
        markers,
        mask=clean_mask
    )

    # ========================================================
    # 6. Characterize detected regions
    # ========================================================

    regions = regionprops(labels)

    bubbles = []

    for region in regions:

        cy, cx = region.centroid
        area = region.area

        radius = np.sqrt(area / np.pi)

        bubbles.append({
            "label": region.label,
            "x": cx,
            "y": cy,
            "area": area,
            "radius": radius
        })

    # ========================================================
    # 7. Return detection products
    # ========================================================

    return {
        "threshold": threshold,
        "bright_mask": bright_mask,
        "clean_mask": clean_mask,
        "distance": distance,
        "coords": coords,
        "markers": markers,
        "labels": labels,
        "bubbles": bubbles
    }
def calculate_physical_extent(shape, spatial_sample):
    """
    Calculate the physical extent of a 2D map centered at (0, 0).

    Parameters
    ----------
    shape : tuple
        Map dimensions (ny, nx).
    spatial_sample : float
        Spatial sampling in Mm/pixel.

    Returns
    -------
    extent : list
        Physical boundaries [xmin, xmax, ymin, ymax] in Mm.
    """
    ny, nx = shape

    return [
        -nx / 2 * spatial_sample,
         nx / 2 * spatial_sample,
        -ny / 2 * spatial_sample,
         ny / 2 * spatial_sample
    ]