"""
Bank ma'lumotlarini tozalash moduli.
Array (list/dict) ustunlarini ham qo'llab-quvvatlaydi - NoSQL uchun.
"""

import pandas as pd
import numpy as np


def _has_unhashable(df, col):
    """Tekshiradi - ustunda list yoki dict qiymatlar bormi (unhashable types)."""
    try:
        sample = df[col].dropna().head(10)
        return any(isinstance(v, (list, dict)) for v in sample)
    except Exception:
        return False


def _safe_duplicated_count(df):
    """Xavfsiz dublikat hisoblash - array ustunlarni hisobga olmaydi."""
    try:
        # Hashable ustunlarni topamiz
        hashable_cols = [c for c in df.columns if not _has_unhashable(df, c)]
        if not hashable_cols:
            return 0
        return df[hashable_cols].duplicated().sum()
    except Exception:
        return 0


def _safe_drop_duplicates(df):
    """Xavfsiz dublikatlarni olib tashlash."""
    try:
        hashable_cols = [c for c in df.columns if not _has_unhashable(df, c)]
        if not hashable_cols:
            return df, 0
        before = len(df)
        # Dublikatlarni hashable ustunlar bo'yicha topamiz va o'chiramiz
        mask = df[hashable_cols].duplicated(keep='first')
        df = df[~mask].reset_index(drop=True)
        removed = before - len(df)
        return df, removed
    except Exception:
        return df, 0


def clean_dataset(df, table_name):
    """Ma'lumotlarni tozalaydi va hisobot qaytaradi."""
    original_len = len(df)
    actions = []
    duplicates_removed = 0
    nulls_filled = 0
    
    # 1. Dublikatlarni olib tashlash (xavfsiz)
    df, dupes = _safe_drop_duplicates(df)
    if dupes > 0:
        duplicates_removed = dupes
        actions.append(f"To'liq dublikatlar o'chirildi: {dupes}")
    
    # 2. Matn ustunlarini normalizatsiya
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    normalized_count = 0
    for col in string_cols:
        # Array ustunlarini o'tkazib yuboramiz
        if _has_unhashable(df, col):
            continue
        try:
            mask = df[col].notna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
                if col in ['first_name', 'last_name']:
                    df.loc[mask, col] = df.loc[mask, col].str.title()
                normalized_count += 1
        except Exception:
            pass
    if normalized_count > 0:
        actions.append(f"Matn ustunlari normalizatsiya qilindi: {normalized_count} ta")
    
    # 3. ID ustunlarida NULL bo'lgan qatorlarni o'chirish
    critical_cols = [c for c in df.columns if c.endswith('_id') and c in df.columns
                      and not _has_unhashable(df, c)]
    if critical_cols:
        before = len(df)
        df = df.dropna(subset=critical_cols).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            actions.append(f"Asosiy maydonlarda NULL bo'lgan qatorlar o'chirildi: {dropped}")
    
    # 4. Nokritik maydonlarda NULL ni to'ldirish
    fill_rules = {
        'region': "Ko'rsatilmagan",
        'phone': 'N/A',
        'description': 'Izohsiz',
        'channel': 'Unknown',
        'branch_code': 'BR000',
        'capitalization': 'AtMaturity',
        'product_type': 'Other',
        'currency': 'UZS',
    }
    for col, fill_val in fill_rules.items():
        if col in df.columns and not _has_unhashable(df, col):
            null_count = df[col].isna().sum()
            if null_count > 0:
                df[col] = df[col].fillna(fill_val)
                nulls_filled += null_count
                actions.append(f"'{col}' ustunidagi NULL to'ldirildi: {null_count} ta ('{fill_val}' qiymati bilan)")
    
    # 5. Sonli ustunlardagi anomaliyalarni tuzatish
    if 'balance' in df.columns:
        try:
            neg_count = (df['balance'] < 0).sum()
            if neg_count > 0:
                df.loc[df['balance'] < 0, 'balance'] = df.loc[df['balance'] < 0, 'balance'].abs()
                actions.append(f"Salbiy balanslar tuzatildi: {neg_count} ta")
        except Exception:
            pass
    
    if 'amount' in df.columns:
        try:
            neg_count = (df['amount'] < 0).sum()
            if neg_count > 0:
                df.loc[df['amount'] < 0, 'amount'] = df.loc[df['amount'] < 0, 'amount'].abs()
                actions.append(f"Salbiy summalar tuzatildi: {neg_count} ta")
            
            null_count = df['amount'].isna().sum()
            if null_count > 0:
                median_val = df['amount'].median()
                df['amount'] = df['amount'].fillna(median_val)
                nulls_filled += null_count
                actions.append(f"'amount' ustunidagi NULL median qiymat bilan to'ldirildi ({median_val:.0f}): {null_count} ta")
        except Exception:
            pass
    
    if 'interest_rate' in df.columns:
        try:
            invalid = ((df['interest_rate'] < 0) | (df['interest_rate'] > 100)).sum()
            if invalid > 0:
                valid_mask = (df['interest_rate'] >= 0) & (df['interest_rate'] <= 100)
                if valid_mask.any():
                    median_rate = df.loc[valid_mask, 'interest_rate'].median()
                    df.loc[~valid_mask & df['interest_rate'].notna(), 'interest_rate'] = median_rate
                    actions.append(f"Noto'g'ri foiz stavkalari tuzatildi: {invalid} ta")
            
            null_count = df['interest_rate'].isna().sum()
            if null_count > 0:
                df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())
                nulls_filled += null_count
        except Exception:
            pass
    
    if 'principal_amount' in df.columns:
        try:
            null_count = df['principal_amount'].isna().sum()
            if null_count > 0:
                df['principal_amount'] = df['principal_amount'].fillna(df['principal_amount'].median())
                nulls_filled += null_count
                actions.append(f"'principal_amount' ustunidagi NULL to'ldirildi: {null_count} ta")
            
            neg = (df['principal_amount'] < 0).sum()
            if neg > 0:
                df.loc[df['principal_amount'] < 0, 'principal_amount'] = \
                    df.loc[df['principal_amount'] < 0, 'principal_amount'].abs()
                actions.append(f"Salbiy kredit summalari tuzatildi: {neg} ta")
        except Exception:
            pass
    
    # 6. Sana ustunlarini datetime turiga o'tkazish
    date_cols = [c for c in df.columns if 'date' in c.lower() and not _has_unhashable(df, c)]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass
    if date_cols:
        actions.append(f"Datetime turiga o'tkazildi: {len(date_cols)} ta ustun")
    
    report = {
        'table': table_name,
        'original_rows': original_len,
        'final_rows': len(df),
        'duplicates_removed': duplicates_removed,
        'nulls_filled': nulls_filled,
        'actions': actions,
    }
    
    return df, report


def get_quality_report(df, table_name=""):
    """Ma'lumot sifati hisobotini yaratadi."""
    total_cells = len(df) * len(df.columns) if len(df) > 0 else 1
    nulls = int(df.isna().sum().sum())
    dupes = int(_safe_duplicated_count(df))
    
    return {
        'table': table_name,
        'rows': len(df),
        'columns': len(df.columns),
        'null_count': nulls,
        'null_pct': round(nulls / total_cells * 100, 2),
        'duplicate_count': dupes,
        'quality_score': round((total_cells - nulls - dupes * len(df.columns)) / total_cells * 100, 2),
    }
