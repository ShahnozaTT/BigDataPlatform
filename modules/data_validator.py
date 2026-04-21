"""
Bank ma'lumotlarini validatsiya qilish moduli.
NoSQL array ustunlarini ham qo'llab-quvvatlaydi.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def _has_unhashable(df, col):
    """Ustunda list yoki dict qiymatlar bormi (NoSQL array)."""
    try:
        if col not in df.columns:
            return False
        sample = df[col].dropna().head(10)
        return any(isinstance(v, (list, dict)) for v in sample)
    except Exception:
        return False


def _safe_duplicated(series):
    """Xavfsiz duplicated — NoSQL arrays uchun."""
    try:
        return series.duplicated().sum()
    except TypeError:
        # list/dict bor — stringga aylantirib tekshiramiz
        return series.astype(str).duplicated().sum()


def validate_dataset(data_dict):
    """Ma'lumotlar to'plamini validatsiya qiladi."""
    checks = []
    
    clients = data_dict.get('clients', pd.DataFrame())
    accounts = data_dict.get('accounts', pd.DataFrame())
    transactions = data_dict.get('transactions', pd.DataFrame())
    loans = data_dict.get('loans', pd.DataFrame())
    deposits = data_dict.get('deposits', pd.DataFrame())
    
    # ===== MIJOZLAR TEKSHIRUVLARI =====
    if not clients.empty:
        if 'client_id' in clients.columns:
            dup_count = _safe_duplicated(clients['client_id'])
            checks.append({
                'rule': "client_id ning noyobligi",
                'status': 'passed' if dup_count == 0 else 'failed',
                'message': f"client_id dublikatlari: {dup_count}"
                           if dup_count == 0 else f"⚠️ client_id dublikatlari topildi: {dup_count}",
            })
        
        if 'inn' in clients.columns and not _has_unhashable(clients, 'inn'):
            try:
                invalid_inn = clients['inn'].dropna().astype(str).apply(
                    lambda x: not (x.isdigit() and len(x) == 9)
                ).sum()
                checks.append({
                    'rule': "INN formati (9 raqam)",
                    'status': 'passed' if invalid_inn == 0 else 'warning',
                    'message': f"Noto'g'ri INN: {invalid_inn}",
                })
            except Exception:
                pass
        
        if 'birth_date' in clients.columns:
            try:
                bd = pd.to_datetime(clients['birth_date'], errors='coerce')
                ages = ((pd.Timestamp.now() - bd).dt.days / 365.25)
                underage = int((ages < 18).sum())
                checks.append({
                    'rule': "Mijoz yoshi ≥ 18",
                    'status': 'passed' if underage == 0 else 'warning',
                    'message': f"18 yoshdan kichik mijozlar: {underage}",
                })
            except Exception:
                pass
    
    # ===== HISOBLAR TEKSHIRUVLARI =====
    if not accounts.empty:
        if 'account_number' in accounts.columns and not _has_unhashable(accounts, 'account_number'):
            try:
                invalid = accounts['account_number'].dropna().astype(str).apply(
                    lambda x: not (x.isdigit() and len(x) == 20)
                ).sum()
                checks.append({
                    'rule': "Hisob raqami formati (20 raqam)",
                    'status': 'passed' if invalid == 0 else 'warning',
                    'message': f"Noto'g'ri hisob raqamlari: {invalid}",
                })
            except Exception:
                pass
        
        if 'client_id' in accounts.columns and 'client_id' in clients.columns:
            try:
                orphans = (~accounts['client_id'].isin(clients['client_id'])).sum()
                checks.append({
                    'rule': "Bog'lanish: accounts → clients",
                    'status': 'passed' if orphans == 0 else 'failed',
                    'message': f"Mijozsiz hisoblar: {orphans}",
                })
            except Exception:
                pass
        
        if 'balance' in accounts.columns:
            try:
                neg = int((accounts['balance'] < 0).sum())
                checks.append({
                    'rule': "Balanslar salbiy emas",
                    'status': 'passed' if neg == 0 else 'warning',
                    'message': f"Salbiy balans: {neg} ta hisob",
                })
            except Exception:
                pass
    
    # ===== OPERATSIYALAR TEKSHIRUVLARI =====
    if not transactions.empty:
        if 'transaction_date' in transactions.columns:
            try:
                dates = pd.to_datetime(transactions['transaction_date'], errors='coerce')
                future = int((dates > pd.Timestamp.now()).sum())
                checks.append({
                    'rule': "Operatsiya sanalari kelajakda emas",
                    'status': 'passed' if future == 0 else 'failed',
                    'message': f"Kelajak sanali operatsiyalar: {future}",
                })
            except Exception:
                pass
        
        if 'amount' in transactions.columns:
            try:
                invalid = int((transactions['amount'] <= 0).sum())
                checks.append({
                    'rule': "Operatsiya summalari musbat",
                    'status': 'passed' if invalid == 0 else 'warning',
                    'message': f"Noto'g'ri summali operatsiyalar: {invalid}",
                })
            except Exception:
                pass
        
        if 'account_id' in transactions.columns and 'account_id' in accounts.columns:
            try:
                orphans = (~transactions['account_id'].isin(accounts['account_id'])).sum()
                checks.append({
                    'rule': "Bog'lanish: transactions → accounts",
                    'status': 'passed' if orphans == 0 else 'warning',
                    'message': f"Hisobsiz operatsiyalar: {orphans}",
                })
            except Exception:
                pass
    
    # ===== KREDITLAR TEKSHIRUVLARI =====
    if not loans.empty:
        if 'interest_rate' in loans.columns:
            try:
                invalid = int(((loans['interest_rate'] < 0) | (loans['interest_rate'] > 100)).sum())
                checks.append({
                    'rule': "Kredit foiz stavkalari [0, 100]% oralig'ida",
                    'status': 'passed' if invalid == 0 else 'warning',
                    'message': f"Noto'g'ri stavkalar: {invalid}",
                })
            except Exception:
                pass
        
        if 'principal_amount' in loans.columns:
            try:
                invalid = int((loans['principal_amount'] <= 0).sum())
                checks.append({
                    'rule': "Kredit summasi musbat",
                    'status': 'passed' if invalid == 0 else 'warning',
                    'message': f"Noto'g'ri summali kreditlar: {invalid}",
                })
            except Exception:
                pass
        
        if 'issue_date' in loans.columns and 'maturity_date' in loans.columns:
            try:
                issue = pd.to_datetime(loans['issue_date'], errors='coerce')
                maturity = pd.to_datetime(loans['maturity_date'], errors='coerce')
                invalid = int((maturity <= issue).sum())
                checks.append({
                    'rule': "To'lov sanasi > berilgan sanadan",
                    'status': 'passed' if invalid == 0 else 'failed',
                    'message': f"Noto'g'ri sanali kreditlar: {invalid}",
                })
            except Exception:
                pass
        
        if 'client_id' in loans.columns and 'client_id' in clients.columns:
            try:
                orphans = (~loans['client_id'].isin(clients['client_id'])).sum()
                checks.append({
                    'rule': "Bog'lanish: loans → clients",
                    'status': 'passed' if orphans == 0 else 'failed',
                    'message': f"Mijozsiz kreditlar: {orphans}",
                })
            except Exception:
                pass
    
    # ===== DEPOZITLAR TEKSHIRUVLARI =====
    if not deposits.empty:
        if 'interest_rate' in deposits.columns:
            try:
                invalid = int(((deposits['interest_rate'] < 0) | (deposits['interest_rate'] > 100)).sum())
                checks.append({
                    'rule': "Depozit stavkalari [0, 100]% oralig'ida",
                    'status': 'passed' if invalid == 0 else 'warning',
                    'message': f"Noto'g'ri depozit stavkalari: {invalid}",
                })
            except Exception:
                pass
        
        if 'client_id' in deposits.columns and 'client_id' in clients.columns:
            try:
                orphans = (~deposits['client_id'].isin(clients['client_id'])).sum()
                checks.append({
                    'rule': "Bog'lanish: deposits → clients",
                    'status': 'passed' if orphans == 0 else 'failed',
                    'message': f"Mijozsiz depozitlar: {orphans}",
                })
            except Exception:
                pass
    
    return {
        'checks': checks,
        'timestamp': datetime.now().isoformat(),
    }
