#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:53:34 2026

@author: daniel
"""

from astropy.io import fits
import numpy as np
from scipy import ndimage

def load_magnetogram_frame(filename, frame_number=0):
    """
    Load one magnetogram frame from a FITS cube.
    """
    with fits.open(
        filename,
        memmap=True,
        do_not_scale_image_data=True
    ) as hdul:
        header = hdul[0].header.copy()
        bscale = header.get("BSCALE", 1.0)
        bzero = header.get("BZERO", 0.0)
        raw = hdul[0].data[frame_number]
        mag_frame = raw.astype(np.float64)
        mag_frame = mag_frame * bscale + bzero
    
    return mag_frame, header


def detect_magnetic_regions(mag_frame, percentile=80, crop=(150, 845)):
    """
    Detect connected magnetic regions in a magnetogram.

    Parameters
    ----------
    mag_frame : np.ndarray
        2D magnetogram.
    percentile : float, optional
        Percentile of |B| used as the detection threshold.
    crop : tuple, optional
        Pixel limits (min, max) used to exclude the image edges.

    Returns
    -------
    regions : list of dict
        Properties of each detected region.
    labeled_regions : np.ndarray
        Array containing the connected-component labels.
    mag_mask : np.ndarray
        Boolean magnetic-field mask.
    threshold : float
        Magnetic-field threshold corresponding to the selected percentile.
    """
    abs_mag = np.abs(mag_frame)
    # Compute threshold using finite values only
    finite_values = abs_mag[np.isfinite(abs_mag)]
    threshold = np.percentile(finite_values, percentile)
    # Magnetic-field mask
    mag_mask = (
        np.isfinite(abs_mag) &
        (abs_mag > threshold)
    )
    # Exclude image edges
    i_min, i_max = crop
    central_mask = np.zeros_like(mag_mask, dtype=bool)
    central_mask[i_min:i_max, i_min:i_max] = True
    mag_mask &= central_mask
    # Identify connected regions using 8-connectivity
    structure = np.ones((3, 3), dtype=int)
    labeled_regions, n_regions = ndimage.label(
        mag_mask,
        structure=structure
    )
    # Characterize regions
    regions = []
    for label_id in range(1, n_regions + 1):
        region_mask = labeled_regions == label_id
        area_pixels = region_mask.sum()
        y_pixels, x_pixels = np.where(region_mask)
        x_center = np.mean(x_pixels)
        y_center = np.mean(y_pixels)
        r_equivalent = np.sqrt(area_pixels / np.pi)

        regions.append({
            "label": label_id,
            "x_center": x_center,
            "y_center": y_center,
            "area_pixels": area_pixels,
            "r_equivalent_pixels": r_equivalent,
            #"mask": region_mask
            })
        # Equivalent-radius statistics
    if regions:
        equivalent_radii = np.array([
            region["r_equivalent_pixels"] for region in regions
        ])

        max_radius = np.max(equivalent_radii)
        mean_radius = np.mean(equivalent_radii)
        std_radius = np.std(equivalent_radii)
        print(f"Magnetic threshold: {threshold:.2f}")
        print(f"Number of detected regions: {len(regions)}")
        print(f"Maximum equivalent radius: {max_radius:.2f} pixels")
        print(f"Mean equivalent radius: {mean_radius:.2f} pixels")
        print(f"Standard deviation: {std_radius:.2f} pixels")
    else:
        print("No magnetic regions detected.")
    return regions, labeled_regions, mag_mask, threshold


def filter_regions_by_radius(
    regions,
    labeled_regions,
    mag_frame,
    min_radius=5
):
    """
    Filter detected regions by equivalent radius and compute
    magnetic-field statistics inside the original region and
    its equivalent circle.

    Parameters
    ----------
    regions : list of dict
        Detected magnetic regions.
    labeled_regions : np.ndarray
        Labeled array containing all connected regions.
    mag_frame : np.ndarray
        2D magnetogram used for the region detection.
    min_radius : float, optional
        Minimum equivalent radius in pixels.

    Returns
    -------
    selected_regions : list of dict
        Selected regions including masks and magnetic-field statistics.
    """

    selected_regions = []

    # Pixel-coordinate arrays
    y, x = np.indices(mag_frame.shape)

    for region in regions:

        if region["r_equivalent_pixels"] >= min_radius:

            label = region["label"]
            x_center = region["x_center"]
            y_center = region["y_center"]
            radius = region["r_equivalent_pixels"]

            # ----------------------------------------------------------
            # Original-region mask
            # ----------------------------------------------------------

            region_mask = labeled_regions == label

            # ----------------------------------------------------------
            # Equivalent-circle mask
            # ----------------------------------------------------------

            circle_mask = (
                (x - x_center)**2 +
                (y - y_center)**2
                <= radius**2
            )

            # ----------------------------------------------------------
            # Magnetic-field values inside the original region
            # ----------------------------------------------------------

            mag_mask_values = mag_frame[
                region_mask & np.isfinite(mag_frame)
            ]

            mean_mag_mask = np.mean(mag_mask_values)
            std_mag_mask = np.std(mag_mask_values)

            # ----------------------------------------------------------
            # Magnetic-field values inside the equivalent circle
            # ----------------------------------------------------------

            mag_circle_values = mag_frame[
                circle_mask & np.isfinite(mag_frame)
            ]

            mean_mag_circle = np.mean(mag_circle_values)
            std_mag_circle = np.std(mag_circle_values)

            # ----------------------------------------------------------
            # Store selected-region information
            # ----------------------------------------------------------

            selected_region = region.copy()

            selected_region.update({
                "mask": region_mask,
                "circle_mask": circle_mask,
                "mean_magnetic_strength_mask": mean_mag_mask,
                "std_magnetic_strength_mask": std_mag_mask,
                "mean_magnetic_strength_circle": mean_mag_circle,
                "std_magnetic_strength_circle": std_mag_circle
            })

            selected_regions.append(selected_region)

    print(
        f"Selected regions (r >= {min_radius} px): "
        f"{len(selected_regions)}"
    )

    return selected_regions