"""OULAD evidence boundaries, adapted from MARS's forecast/utility separation.

Unlike topic odds in xes-mars, course risk is a scalar and must never be
normalized into invented topic probabilities. These helpers only select
observable recommendation evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENROLLMENT = ['id_student', 'code_module', 'code_presentation']


def finite_number(value, name, *, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f'Invalid {name}')
    return number


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')


def numeric_column(frame, column, *, optional=False):
    values = pd.to_numeric(frame[column], errors='raise').astype(float)
    if optional:
        invalid = values.notna() & ~np.isfinite(values)
    else:
        invalid = ~np.isfinite(values)
    if invalid.any():
        raise ValueError(f'Invalid {column}')
    return values


def observable_context(resources, interactions, enrollment, as_of_day):
    """Return eligible catalog and positive resource-day activity before cutoff.

    week_from is a conservative planned-week gate, not proof of release.
    An explicit available_from_day overrides that gate. Bounds are inclusive
    for availability, exclusive for interaction history.
    """
    day = finite_number(as_of_day, 'as_of_day')
    if day != int(day):
        raise ValueError('as_of_day must be an integer course day')
    if len(enrollment) != 3 or any(pd.isna(v) for v in enrollment):
        raise ValueError('Expected one complete enrollment identity')
    require_columns(resources, ['id_site', 'code_module', 'code_presentation', 'activity_type'])
    require_columns(interactions, ENROLLMENT + ['id_site', 'date', 'sum_click'])
    _, module, presentation = enrollment
    catalog = resources.loc[
        (resources['code_module'] == module) & (resources['code_presentation'] == presentation)
    ].copy()
    catalog['id_site'] = numeric_column(catalog, 'id_site')
    if (catalog['id_site'] < 0).any() or (catalog['id_site'] % 1 != 0).any():
        raise ValueError('id_site must be a nonnegative integer')
    catalog['id_site'] = catalog['id_site'].astype('int64')
    if catalog['id_site'].duplicated().any():
        raise ValueError('Duplicate resource identity within module-presentation')
    if catalog['activity_type'].isna().any() or catalog['activity_type'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing activity_type')
    catalog['activity_type'] = catalog['activity_type'].astype(str)
    for column in ['week_from', 'week_to', 'available_from_day', 'available_until_day']:
        catalog[column] = numeric_column(catalog, column, optional=True) if column in catalog else np.nan
    explicit = catalog['available_from_day']
    start = explicit.where(explicit.notna(), catalog['week_from'] * 7)
    end = catalog['available_until_day']
    if ((start > end) & start.notna() & end.notna()).any():
        raise ValueError('Availability ends before it starts')
    # Historical evidence survives expiry and schedule changes. Actual prior
    # access is evidence even when the resource is no longer a candidate.
    history_metadata = catalog[['id_site', 'activity_type']].copy()
    catalog = catalog.loc[(start.isna() | (start <= day)) & (end.isna() | (end >= day))]
    catalog = catalog.sort_values('id_site').reset_index(drop=True)

    selected = interactions.loc[
        (interactions['id_student'] == enrollment[0])
        & (interactions['code_module'] == module)
        & (interactions['code_presentation'] == presentation),
        ['id_site', 'date', 'sum_click'],
    ].copy()
    selected['date'] = numeric_column(selected, 'date')
    if (selected['date'] % 1 != 0).any():
        raise ValueError('date must use integer course days')
    selected = selected.loc[selected['date'] < day].copy()
    selected['sum_click'] = numeric_column(selected, 'sum_click')
    if (selected['sum_click'] < 0).any():
        raise ValueError('sum_click cannot be negative')
    selected['id_site'] = numeric_column(selected, 'id_site')
    if (selected['id_site'] < 0).any() or (selected['id_site'] % 1 != 0).any():
        raise ValueError('id_site must be a nonnegative integer')
    selected['id_site'] = selected['id_site'].astype('int64')
    observed = (selected.groupby(['id_site', 'date'], as_index=False, sort=True)['sum_click'].sum())
    observed = observed.loc[observed['sum_click'] > 0]
    observed = observed.merge(history_metadata, on='id_site', how='inner', validate='many_to_one')
    return catalog, observed
