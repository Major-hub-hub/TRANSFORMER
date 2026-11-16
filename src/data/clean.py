'''Load and clean NSL-KDD records for downstream modeling.'''

from __future__ import annotations

from pathlib import Path

import pandas as pd

NSL_KDD_COLUMNS = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'label',
    'difficulty',
]

ATTACK_FAMILY_MAP = {
    'normal': 'Normal',
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',
    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'httptunnel': 'R2L',
    'worm': 'R2L',
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'xterm': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
}


def load_raw(path: str | Path) -> pd.DataFrame:
    '''Load raw NSL-KDD text files and apply the canonical 43-column schema.'''
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)

    if df.isna().to_numpy().any():
        raise AssertionError('Raw data contains nulls; expected fully non-null rows.')

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        raise AssertionError(f'Raw data contains {duplicate_count} duplicate rows.')

    return df


def drop_degenerate(df: pd.DataFrame) -> pd.DataFrame:
    '''Drop constant columns and remove metadata fields not meant as model features.'''
    protected = {'label', 'binary_label', 'attack_family'}
    nunique = df.nunique(dropna=False)
    constant_columns = [
        column for column, value_count in nunique.items() if value_count <= 1 and column not in protected
    ]

    columns_to_drop = set(constant_columns)
    columns_to_drop.add('difficulty')
    final_drop_list = [column for column in columns_to_drop if column in df.columns]

    if not final_drop_list:
        return df.copy()

    return df.drop(columns=final_drop_list)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    '''Convert categoricals to category dtype and integer features to float32.'''
    out = df.copy()

    categorical_columns = ['protocol_type', 'service', 'flag', 'label', 'attack_family']
    for column in categorical_columns:
        if column in out.columns:
            out[column] = out[column].astype('category')

    integer_columns = [
        column
        for column in out.columns
        if pd.api.types.is_integer_dtype(out[column]) and column != 'binary_label'
    ]
    if integer_columns:
        out[integer_columns] = out[integer_columns].astype('float32')

    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    '''Add binary and 5-family target columns derived from the raw attack label.'''
    if 'label' not in df.columns:
        raise KeyError('Expected a label column before generating targets.')

    out = df.copy()
    normalized_label = out['label'].astype(str).str.strip().str.rstrip('.').str.lower()

    out['binary_label'] = (normalized_label != 'normal').astype('int8')
    out['attack_family'] = normalized_label.map(ATTACK_FAMILY_MAP)

    unknown_labels = sorted(normalized_label[out['attack_family'].isna()].unique())
    if unknown_labels:
        raise ValueError(f'Unmapped attack labels found: {unknown_labels}')

    out['attack_family'] = pd.Categorical(
        out['attack_family'],
        categories=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
        ordered=False,
    )

    return out
