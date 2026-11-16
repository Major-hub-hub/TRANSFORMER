'''Fit and apply leakage-safe feature transformations for NSL-KDD.'''

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMNS = {'label', 'binary_label', 'attack_family'}
CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']
RATE_SUFFIX = '_rate'
CORE_HEAVY_TAIL_COLUMNS = {'duration', 'src_bytes', 'dst_bytes'}


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f'Missing required columns: {missing_columns}')


def _one_hot(values: np.ndarray, categories: list[str]) -> np.ndarray:
    if not categories:
        return np.empty((values.shape[0], 0), dtype=np.float32)

    encoded = np.zeros((values.shape[0], len(categories)), dtype=np.float32)
    for category_index, category_name in enumerate(categories):
        encoded[:, category_index] = (values == category_name).astype(np.float32)

    return encoded


def fit_transformers(train_df: pd.DataFrame) -> dict[str, Any]:
    '''Fit category encoders and numeric normalizers using training data only.'''
    _require_columns(train_df, CATEGORICAL_COLUMNS)

    feature_df = train_df.drop(columns=[column for column in TARGET_COLUMNS if column in train_df.columns])

    numeric_columns = [
        column
        for column in feature_df.columns
        if column not in CATEGORICAL_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[column])
    ]

    rate_columns = [column for column in numeric_columns if column.endswith(RATE_SUFFIX)]
    heavy_tailed_columns = [
        column
        for column in numeric_columns
        if column not in rate_columns
        and (
            column in CORE_HEAVY_TAIL_COLUMNS
            or 'count' in column
            or column.startswith('num_')
        )
    ]
    passthrough_numeric_columns = [
        column
        for column in numeric_columns
        if column not in rate_columns and column not in heavy_tailed_columns
    ]

    protocol_categories = pd.Index(feature_df['protocol_type'].astype(str)).unique().tolist()
    flag_categories = pd.Index(feature_df['flag'].astype(str)).unique().tolist()

    service_categories = pd.Index(feature_df['service'].astype(str)).unique().tolist()
    service_to_index = {service_name: idx for idx, service_name in enumerate(service_categories, start=1)}

    heavy_center = None
    heavy_scale = None
    if heavy_tailed_columns:
        heavy_values = feature_df[heavy_tailed_columns].astype('float32').clip(lower=0.0)
        heavy_values = np.log1p(heavy_values)
        heavy_center = heavy_values.mean(axis=0).to_numpy(dtype=np.float32)
        heavy_scale = heavy_values.std(axis=0, dodo=0).to_numpy(dtype=np.float32)
        heavy_scale = np.where(heavy_scale == 0.0, 1.0, heavy_scale).astype(np.float32)

    feature_names = []
    feature_names.extend([f'protocol_type={value}' for value in protocol_categories])
    feature_names.extend([f'flag={value}' for value in flag_categories])
    feature_names.append('service_index')
    feature_names.extend([f'{column}_z' for column in heavy_tailed_columns])
    feature_names.extend(rate_columns)
    feature_names.extend(passthrough_numeric_columns)

    return {
        'protocol_categories': protocol_categories,
        'flag_categories': flag_categories,
        'service_to_index': service_to_index,
        'service_vocab_size': len(service_to_index) + 1,
        'rate_columns': rate_columns,
        'heavy_tailed_columns': heavy_tailed_columns,
        'passthrough_numeric_columns': passthrough_numeric_columns,
        'heavy_center': heavy_center,
        'heavy_scale': heavy_scale,
        'feature_names': feature_names,
    }


def transform(df: pd.DataFrame, fitted: Mapping[str, Any]) -> np.ndarray:
    '''Apply fitted feature transformations and return an (N, F) float32 matrix.'''
    _require_columns(df, CATEGORICAL_COLUMNS)

    protocol_categories = list(fitted.get('protocol_categories', []))
    flag_categories = list(fitted.get('flag_categories', []))
    service_to_index = dict(fitted.get('service_to_index', {}))
    rate_columns = list(fitted.get('rate_columns', []))
    heavy_tailed_columns = list(fitted.get('heavy_tailed_columns', []))
    passthrough_numeric_columns = list(fitted.get('passthrough_numeric_columns', []))
    heavy_center = fitted.get('heavy_center')
    heavy_scale = fitted.get('heavy_scale')

    _require_columns(df, rate_columns + heavy_tailed_columns + passthrough_numeric_columns)

    blocks: list[np.ndarray] = []

    protocol_values = df['protocol_type'].astype(str).to_numpy()
    flag_values = df['flag'].astype(str).to_numpy()
    blocks.append(_one_hot(protocol_values, protocol_categories))
    blocks.append(_one_hot(flag_values, flag_categories))

    service_index = (
        df['service'].astype(str).map(service_to_index).fillna(0).astype('int32').to_numpy().reshape(-1, 1)
    )
    blocks.append(service_index.astype(np.float32))

    if heavy_tailed_columns:
        if heavy_center is None or heavy_scale is None:
            raise ValueError('fitted is missing heavy_center/heavy_scale for heavy-tailed columns.')

        heavy_values = df[heavy_tailed_columns].astype('float32').clip(lower=0.0)
        heavy_values = np.log1p(heavy_values)
        heavy_array = heavy_values.to_numpy(dtype=np.float32, copy=True)
        heavy_array = (heavy_array - np.asarray(heavy_center, dtype=np.float32)) / np.asarray(
            heavy_scale,
            dtype=np.float32,
        )
        blocks.append(heavy_array)

    if rate_columns:
        rate_array = df[rate_columns].to_numpy(dtype=np.float32, copy=True)
        blocks.append(rate_array)

    if passthrough_numeric_columns:
        passthrough_array = df[passthrough_numeric_columns].to_numpy(dtype=np.float32, copy=True)
        blocks.append(passthrough_array)

    if not blocks:
        return np.empty((len(df), 0), dtype=np.float32)

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
