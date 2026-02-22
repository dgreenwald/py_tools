#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def pairwise_match(treated_score, untreated_score, caliper=0.0, replacement=True):
    """One-to-one matching based on propensity scores.

    For each treated unit, finds the closest untreated unit by absolute
    difference in propensity score. An optional caliper can be used to
    exclude matches that are too distant.

    Parameters
    ----------
    treated_score : array-like of shape (n_treated,)
        Propensity scores for the treated group.
    untreated_score : array-like of shape (n_control,)
        Propensity scores for the control (untreated) group.
    caliper : float, optional
        Maximum allowed absolute difference in propensity score. Treated
        units with no control unit within the caliper are excluded from the
        result. A value of ``0.0`` (default) means no caliper is applied.
    replacement : bool, optional
        Whether control units can be used as a match more than once.
        Currently only ``True`` (matching with replacement) is supported.
        Defaults to ``True``.

    Returns
    -------
    good_rows : ndarray of bool, shape (n_treated,)
        Boolean mask indicating which treated units were successfully matched
        (i.e. had at least one control unit within the caliper).
    min_ix : ndarray of int, shape (n_matched,)
        Indices into *untreated_score* giving the best-matching control unit
        for each matched treated unit (rows where *good_rows* is ``True``).

    Raises
    ------
    NotImplementedError
        If ``replacement=False`` (not yet implemented).
    """

    # Get differences for all combinations
    dist_mat = np.abs(treated_score[:, np.newaxis] - untreated_score[np.newaxis, :])

    # Apply caliper if specified
    if caliper > 0.0:
        ix = dist_mat > caliper
        dist_mat[ix] = np.inf

    # Keep only rows that have an acceptable solution
    good_rows = np.any(np.isfinite(dist_mat), axis=1)
    dist_mat = dist_mat[good_rows, :]

    if replacement:
        min_ix = np.argmin(dist_mat, axis=1)

    else:
        # For now, only works with replacement
        raise NotImplementedError(
            "pairwise_match with replacement=False is not implemented."
        )

    return good_rows, min_ix
