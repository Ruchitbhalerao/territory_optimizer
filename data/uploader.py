"""
Excel / CSV uploader for actual data files.

Reads dealer, FTC, and F2D (relationship) data from an uploaded file,
maps columns to the internal format, and saves as parquet files for the
pipeline to consume.
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEALER_COLUMN_MAP = {
    'dealer_id': 'dealer_id',
    'SM_id': 'sm_id',
    'dealer_type_static_mobile': 'dealer_type',
    'dealer_type': 'dealer_type',
    'AvgCasesPerDay': 'average_cases_per_day',
    'average_cases_per_day': 'average_cases_per_day',
    'Count_BFL_Disbursement': 'count_bfl_disbursement',
    'count_bfl_disbursement': 'count_bfl_disbursement',
    'dealer_latitude': 'dealer_latitude',
    'dealer_longitude': 'dealer_longitude',
    'latitude': 'dealer_latitude',
    'lat': 'dealer_latitude',
    'longitude': 'dealer_longitude',
    'lon': 'dealer_longitude',
    'lng': 'dealer_longitude',
}

FTC_COLUMN_MAP = {
    'FTC_ID': 'ftc_id',
    'ftc_id': 'ftc_id',
    'FTE_ID': 'sm_id',
    'sm_id': 'sm_id',
    'Count_BFL_Disbursement': 'count_bfl_disbursement',
    'count_bfl_disbursement': 'count_bfl_disbursement',
    'PER_SUM_MOB03': 'per_sum_mob',
    'per_sum_mob': 'per_sum_mob',
    'AvgCasesPerDay': 'average_cases_per_day',
    'average_cases_per_day': 'average_cases_per_day',
    'FTC_Vintage': 'ftc_vintage',
    'ftc_vintage': 'ftc_vintage',
    'NTB_share_per': 'ntb_share',
    'ntb_share': 'ntb_share',
    'Cross_Sell_share_per': 'cross_sell',
    'cross_sell': 'cross_sell',
}

F2D_COLUMN_MAP = {
    'FTC_ID': 'ftc_id',
    'ftc_id': 'ftc_id',
    'dealer_id': 'dealer_id',
    'AvgCasesPerDay': 'avg_cases_per_day',
    'avg_cases_per_day': 'avg_cases_per_day',
}


def _normalize_col(name: str) -> str:
    """Normalize a column name for fuzzy matching."""
    n = str(name).strip().lower()
    n = re.sub(r'[\s_\-\.]+', '_', n)
    n = re.sub(r'[^a-z0-9_]', '', n)
    n = n.strip('_')
    return n


def _rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    renamed = {}
    norm_lookup = {}
    for excel_col, internal_col in mapping.items():
        norm = _normalize_col(excel_col)
        norm_lookup[norm] = internal_col

    for col in df.columns:
        stripped = str(col).strip()
        if stripped in mapping:
            renamed[col] = mapping[stripped]
            continue
        low = stripped.lower()
        found = False
        for excel_col, internal_col in mapping.items():
            if low == excel_col.lower():
                renamed[col] = internal_col
                found = True
                break
        if found:
            continue
        norm = _normalize_col(stripped)
        if norm in norm_lookup:
            renamed[col] = norm_lookup[norm]
            continue
        for norm_key, internal_col in norm_lookup.items():
            nk_flat = norm_key.replace('_', '')
            n_flat = norm.replace('_', '')
            if nk_flat in n_flat or n_flat in nk_flat:
                renamed[col] = internal_col
                break

    return df.rename(columns=renamed)


def _clean_numeric(val):
    if val is None:
        return np.nan
    if isinstance(val, str):
        val = val.strip()
        if val == '' or val == '-':
            return np.nan
        val = re.sub(r'[^\d.\-]', '', val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


class ExcelUploader:
    """Handles uploading and transforming Excel data files."""

    REQUIRED_DEALER_COLS = [
        'dealer_id', 'sm_id', 'dealer_type', 'product_group',
        'count_bfl_disbursement', 'average_cases_per_day',
        'dealer_latitude', 'dealer_longitude'
    ]
    REQUIRED_FTC_COLS = [
        'ftc_id', 'sm_id', 'product_group', 'ftc_vintage',
        'count_bfl_disbursement', 'average_cases_per_day',
        'per_sum_mob', 'ntb_share', 'cross_sell'
    ]
    REQUIRED_REL_COLS = [
        'dealer_id', 'ftc_id', 'product_category', 'avg_cases_per_day'
    ]

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, file_path: str) -> dict:
        """
        Read an Excel file with 3 sheets (or 3 separate files) and convert
        to internal parquet format.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.xlsx':
            return self._process_excel(path)
        elif suffix == '.zip':
            return self._process_zip(path)
        elif suffix == '.csv':
            return self._process_csvs(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _process_excel(self, path: Path) -> dict:
        xls = pd.ExcelFile(path)
        logger.info(f"Excel sheets found: {xls.sheet_names}")

        dealer_sheet = self._find_sheet(xls.sheet_names, ['dealer', 'dealer dataset'], 'dealers')
        ftc_sheet = self._find_sheet(xls.sheet_names, ['ftc', 'ftc dataset', 'ftc data'], 'FTCs')
        f2d_sheet = self._find_sheet(xls.sheet_names, ['f2d', 'relationship', 'f2d dataset', 'f2d data', 'assigned'], 'relationships')

        dealers_raw = pd.read_excel(path, sheet_name=dealer_sheet) if dealer_sheet else pd.DataFrame()
        ftcs_raw = pd.read_excel(path, sheet_name=ftc_sheet) if ftc_sheet else pd.DataFrame()
        rels_raw = pd.read_excel(path, sheet_name=f2d_sheet) if f2d_sheet else pd.DataFrame()

        if not dealer_sheet:
            logger.warning(f"No dealer sheet found. Available sheets: {xls.sheet_names}")
        if not ftc_sheet:
            logger.warning(f"No FTC sheet found. Available sheets: {xls.sheet_names}")
        if not f2d_sheet:
            logger.warning(f"No relationship sheet found. Available sheets: {xls.sheet_names}")

        return self._transform_and_save(dealers_raw, ftcs_raw, rels_raw)

    def _process_zip(self, path: Path) -> dict:
        import zipfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(tmpdir)
            csv_files = list(Path(tmpdir).rglob('*.csv'))

            dealers_raw = self._read_csv_by_pattern(csv_files, ['dealer'])
            ftcs_raw = self._read_csv_by_pattern(csv_files, ['ftc', 'ftc dataset'])
            rels_raw = self._read_csv_by_pattern(csv_files, ['f2d', 'relationship', 'f2d dataset'])

            return self._transform_and_save(dealers_raw, ftcs_raw, rels_raw)

    def _process_csvs(self, path: Path) -> dict:
        df = pd.read_csv(path)
        cols_lower = {c.lower() for c in df.columns}

        if 'sm_id' in cols_lower or 'dealer_type' in cols_lower or 'dealer_type_static_mobile' in cols_lower:
            logger.info(f"Detected {path.name} as dealer data")
            return self._transform_and_save(df, pd.DataFrame(), pd.DataFrame())
        elif 'fte_id' in cols_lower or 'ftc_vintage' in cols_lower or 'per_sum_mob' in cols_lower or 'per_sum_mob03' in cols_lower:
            logger.info(f"Detected {path.name} as FTC data")
            return self._transform_and_save(pd.DataFrame(), df, pd.DataFrame())
        elif len(cols_lower) <= 4 and 'ftc_id' in cols_lower and 'dealer_id' in cols_lower:
            logger.info(f"Detected {path.name} as F2D relationship data")
            return self._transform_and_save(pd.DataFrame(), pd.DataFrame(), df)
        else:
            raise ValueError(f"Cannot determine sheet type for {path.name}: columns={list(df.columns)}")

    def _read_csv_by_pattern(self, csv_files: list, patterns: list):
        for f in csv_files:
            name_lower = f.stem.lower()
            for pat in patterns:
                if pat in name_lower:
                    return pd.read_csv(f)
        return pd.DataFrame()

    def _find_sheet(self, sheet_names: list, patterns: list, label: str = "") -> Optional[str]:
        norm_names = {s: _normalize_col(s) for s in sheet_names}
        for name, nname in norm_names.items():
            for pat in patterns:
                ppat = _normalize_col(pat)
                if ppat in nname or nname in ppat:
                    logger.info(f"Matched sheet '{name}' for {label or patterns}")
                    return name
        logger.warning(f"No sheet matched for {label or patterns} among {sheet_names}")
        return None

    def _transform_and_save(self, dealers_raw: pd.DataFrame,
                            ftcs_raw: pd.DataFrame,
                            rels_raw: pd.DataFrame) -> dict:
        stats = {}

        if not dealers_raw.empty:
            dealers = self._transform_dealers(dealers_raw)
            stats['dealers'] = len(dealers)
            logger.info(f"Uploaded {len(dealers)} dealers")
        else:
            dealers = self._empty_dealers()
            logger.warning("No dealer sheet found — writing empty dealers.parquet")
        dealers.to_parquet(self.output_dir / "dealers.parquet", index=False)

        if not ftcs_raw.empty:
            ftcs = self._transform_ftcs(ftcs_raw)
            stats['ftcs'] = len(ftcs)
            logger.info(f"Uploaded {len(ftcs)} FTCs")
        else:
            ftcs = self._empty_ftcs()
            logger.warning("No FTC sheet found — writing empty ftcs.parquet")
        ftcs.to_parquet(self.output_dir / "ftcs.parquet", index=False)

        if not rels_raw.empty:
            rels = self._transform_relationships(rels_raw)
            stats['relationships'] = len(rels)
            logger.info(f"Uploaded {len(rels)} relationships")
        else:
            rels = self._empty_relationships()
            logger.warning("No relationship sheet found — writing empty relationships.parquet")

        valid_dealers = set(dealers['dealer_id']) if not dealers.empty else set()
        valid_ftcs = set(ftcs['ftc_id']) if not ftcs.empty else set()
        before = len(rels)
        if valid_dealers and valid_ftcs:
            rels = rels[rels['dealer_id'].isin(valid_dealers) & rels['ftc_id'].isin(valid_ftcs)]
        elif valid_dealers:
            rels = rels[rels['dealer_id'].isin(valid_dealers)]
        elif valid_ftcs:
            rels = rels[rels['ftc_id'].isin(valid_ftcs)]
        if len(rels) < before:
            logger.warning(f"Removed {before - len(rels)} orphaned relationships")
        stats['relationships'] = len(rels)
        rels.to_parquet(self.output_dir / "relationships.parquet", index=False)

        pd.DataFrame(columns=[
            'dealer_id', 'related_dealer_id', 'product_group',
            'latitude', 'longitude', 'spatial_distance'
        ]).to_parquet(self.output_dir / "proximity.parquet", index=False)

        return stats

    @staticmethod
    def _empty_dealers() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'dealer_id', 'sm_id', 'dealer_type', 'product_group',
            'count_bfl_disbursement', 'average_cases_per_day',
            'dealer_latitude', 'dealer_longitude'
        ])

    @staticmethod
    def _empty_ftcs() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'ftc_id', 'sm_id', 'product_group', 'ftc_vintage',
            'count_bfl_disbursement', 'average_cases_per_day',
            'per_sum_mob', 'ntb_share', 'cross_sell'
        ])

    @staticmethod
    def _empty_relationships() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'dealer_id', 'ftc_id', 'product_category', 'avg_cases_per_day'
        ])

    def _transform_dealers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _rename_columns(df, DEALER_COLUMN_MAP)

        if 'dealer_id' in df.columns:
            df['dealer_id'] = df['dealer_id'].astype(str)

        if 'dealer_type' in df.columns:
            df['dealer_type'] = df['dealer_type'].astype(str).str.strip().str.lower()
        else:
            df['dealer_type'] = 'mobile'

        if 'sm_id' not in df.columns:
            df['sm_id'] = 'UNKNOWN'
        else:
            df['sm_id'] = df['sm_id'].fillna('UNKNOWN').astype(str)

        for col in ['count_bfl_disbursement', 'average_cases_per_day']:
            if col in df.columns:
                df[col] = df[col].apply(_clean_numeric).fillna(0)

        if 'dealer_latitude' in df.columns:
            df['dealer_latitude'] = pd.to_numeric(df['dealer_latitude'], errors='coerce')
        else:
            df['dealer_latitude'] = np.nan

        if 'dealer_longitude' in df.columns:
            df['dealer_longitude'] = pd.to_numeric(df['dealer_longitude'], errors='coerce')
        else:
            df['dealer_longitude'] = np.nan

        before = len(df)
        df = df.dropna(subset=['dealer_latitude', 'dealer_longitude'])
        if len(df) < before:
            logger.warning(f"Dropped {before - len(df)} dealers with missing coordinates")

        if 'product_group' not in df.columns:
            df['product_group'] = 'personal_loan'

        return df[self.REQUIRED_DEALER_COLS]

    def _transform_ftcs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _rename_columns(df, FTC_COLUMN_MAP)

        if 'ftc_id' in df.columns:
            df['ftc_id'] = df['ftc_id'].astype(str)

        if 'sm_id' not in df.columns:
            df['sm_id'] = 'UNKNOWN'
        else:
            df['sm_id'] = df['sm_id'].fillna('UNKNOWN').astype(str)

        for col in ['count_bfl_disbursement', 'average_cases_per_day',
                     'per_sum_mob', 'ftc_vintage']:
            if col in df.columns:
                df[col] = df[col].apply(_clean_numeric).fillna(0)
            else:
                df[col] = 0.0

        for col in ['ntb_share', 'cross_sell']:
            if col in df.columns:
                df[col] = df[col].apply(_clean_numeric).fillna(0)
                if df[col].max() > 1.0:
                    df[col] = df[col] / 100.0
            else:
                df[col] = 0.0

        if 'product_group' not in df.columns:
            df['product_group'] = 'personal_loan'

        return df[self.REQUIRED_FTC_COLS]

    def _transform_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _rename_columns(df, F2D_COLUMN_MAP)

        if 'dealer_id' in df.columns:
            df['dealer_id'] = df['dealer_id'].astype(str)

        if 'ftc_id' in df.columns:
            df['ftc_id'] = df['ftc_id'].astype(str)

        if 'avg_cases_per_day' in df.columns:
            df['avg_cases_per_day'] = df['avg_cases_per_day'].apply(_clean_numeric).fillna(0)
        else:
            df['avg_cases_per_day'] = 0.0

        if 'product_category' not in df.columns:
            df['product_category'] = 'personal_loan'

        return df[self.REQUIRED_REL_COLS]
