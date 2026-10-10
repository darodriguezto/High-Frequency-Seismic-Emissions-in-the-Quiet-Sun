#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 23:14:32 2026

@author: daniel
"""

"""Track selected magnetic pores across reference magnetograms.

Each reference is expected to contain:
    frame: original cube frame index
    magnetogram: 2D magnetic-field array
    selected_regions: list of detected region dictionaries

Each selected region must contain:
    label, x_center, y_center, r_equivalent_pixels, area_pixels, mask
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def get_pore_properties(ref, region):
    """Extract spatial and signed magnetic-field statistics for one detection.

    Magnetic statistics are computed over the original detected region mask.
    """
    mag_frame = ref["magnetogram"]
    region_mask = region["mask"]
    values = mag_frame[region_mask & np.isfinite(mag_frame)]

    return {
        "frame": ref["frame"],
        "label": region["label"],
        "x_center": region["x_center"],
        "y_center": region["y_center"],
        "r_equivalent_pixels": region["r_equivalent_pixels"],
        "area_pixels": region["area_pixels"],
        "mean_magnetic_field": float(np.mean(values)) if values.size else np.nan,
        "std_magnetic_field": float(np.std(values)) if values.size else np.nan,
    }


def track_pores(magnetograms_of_reference, alpha=2.06, verbose=True):
    """Associate selected pores between successive reference magnetograms.

    Candidate matches satisfy:
        distance_pixels < alpha * previous_radius_pixels + frame_difference / 100

    The assignment minimizes centroid distance, with dummy columns allowing
    unmatched previous pores. Tracks cannot bridge a missing detection.

    Parameters
    ----------
    magnetograms_of_reference : sequence of dict
        Reference magnetograms, in strictly increasing frame order.
    alpha : float, optional
        Scaling factor for the previous pore's equivalent radius.
    verbose : bool, optional
        Print a summary for each comparison.

    Returns
    -------
    dict
        Mapping track_id -> list of detection-property dictionaries.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite, non-negative number")

    if not magnetograms_of_reference:
        return {}

    frames = [ref["frame"] for ref in magnetograms_of_reference]
    if any(current <= previous for previous, current in zip(frames, frames[1:])):
        raise ValueError("Reference magnetograms must be in strictly increasing frame order")

    pore_tracks = {}
    next_track_id = 0

    first_ref = magnetograms_of_reference[0]
    previous_track_ids = {}

    # Initialize a track for every selected pore in the first reference frame.
    for j, region in enumerate(first_ref["selected_regions"]):
        track_id = next_track_id
        next_track_id += 1
        pore_tracks[track_id] = [get_pore_properties(first_ref, region)]
        previous_track_ids[j] = track_id

    # Compare successive elements of the reference list.
    for i in range(len(magnetograms_of_reference) - 1):
        ref_previous = magnetograms_of_reference[i]
        ref_current = magnetograms_of_reference[i + 1]
        pores_previous = ref_previous["selected_regions"]
        pores_current = ref_current["selected_regions"]
        frame_difference = ref_current["frame"] - ref_previous["frame"]

        centroids_previous = np.array(
            [[r["x_center"], r["y_center"]] for r in pores_previous]
        ).reshape(-1, 2)
        centroids_current = np.array(
            [[r["x_center"], r["y_center"]] for r in pores_current]
        ).reshape(-1, 2)

        distance_matrix = cdist(centroids_previous, centroids_current)
        previous_radii = np.array(
            [r["r_equivalent_pixels"] for r in pores_previous]
        )
        distance_thresholds = alpha * previous_radii + frame_difference / 100
        valid_matches = distance_matrix < distance_thresholds[:, None]
        current_track_ids = {}

        if len(pores_previous) > 0 and len(pores_current) > 0:
            n_previous, n_current = distance_matrix.shape
            dummy_cost = max(float(np.max(distance_thresholds)), 1.0) + 1.0

            # Dummy columns permit each previous pore to remain unmatched.
            assignment_cost = np.full(
                (n_previous, n_current + n_previous), dummy_cost
            )
            assignment_cost[:, :n_current] = np.where(
                valid_matches, distance_matrix, dummy_cost * 2
            )

            row_indices, col_indices = linear_sum_assignment(assignment_cost)
            for j, k in zip(row_indices, col_indices):
                if k >= n_current or not valid_matches[j, k]:
                    continue

                track_id = previous_track_ids[j]
                pore_tracks[track_id].append(
                    get_pore_properties(ref_current, pores_current[k])
                )
                current_track_ids[k] = track_id

        # Unmatched current pores start new tracks.
        for k, region in enumerate(pores_current):
            if k not in current_track_ids:
                track_id = next_track_id
                next_track_id += 1
                pore_tracks[track_id] = [get_pore_properties(ref_current, region)]
                current_track_ids[k] = track_id

        previous_track_ids = current_track_ids
        if verbose:
            print(
                f'Frames {ref_previous["frame"]} → {ref_current["frame"]}: '
                f'{len(current_track_ids)} pores in current frame'
            )

    return pore_tracks
def select_persistent_pores(pore_tracks, min_detections=4):
    """
    Select persistent pores and rank them by absolute mean magnetic field.

    Parameters
    ----------
    pore_tracks : dict
        Dictionary containing pore tracking histories.

    min_detections : int, optional
        Minimum number of detections required for a pore
        to be considered persistent.

    Returns
    -------
    sorted_persistent_pores : list of dict
        Persistent pores sorted by the absolute value of their
        time-averaged mean magnetic field (descending order).

        Each dictionary contains:
        - pore_id: ranking position, starting at 1.
        - track_id: original tracking identifier.
        - mean_magnetic_field: signed mean over all detections.
        - history: complete detection history.
    """

    if min_detections < 1:
        raise ValueError("min_detections must be at least 1.")

    sorted_persistent_pores = []

    for track_id, history in pore_tracks.items():

        if len(history) < min_detections:
            continue

        # Compute the mean of the magnetic-field means
        # across all detections of this pore.
        mean_B = np.mean([
            obs["mean_magnetic_field"]
            for obs in history
        ])

        sorted_persistent_pores.append({
            "track_id": track_id,
            "mean_magnetic_field": float(mean_B),
            "history": history
        })

    # Rank by absolute mean magnetic field
    sorted_persistent_pores.sort(
        key=lambda pore: abs(pore["mean_magnetic_field"]),
        reverse=True
    )

    # Assign a new ID according to the ranking
    for rank, pore in enumerate(sorted_persistent_pores, start=1):
        pore["pore_id"] = rank

    return sorted_persistent_pores