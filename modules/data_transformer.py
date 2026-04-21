"""
Модуль трансформации данных и расчёта банковских KPI.
"""

import pandas as pd
import numpy as np


def calculate_kpis(data_dict):
    """Рассчитывает ключевые банковские показатели."""
    clients = data_dict.get('clients', pd.DataFrame())
    accounts = data_dict.get('accounts', pd.DataFrame())
    transactions = data_dict.get('transactions', pd.DataFrame())
    loans = data_dict.get('loans', pd.DataFrame())
    deposits = data_dict.get('deposits', pd.DataFrame())
    
    kpis = {}
    
    # === Кредитные показатели ===
    total_loans = len(loans)
    loan_volume = loans['principal_amount'].sum() if 'principal_amount' in loans.columns else 0
    
    # NPL ratio = NPL loans / Total loans
    if 'status' in loans.columns and total_loans > 0:
        npl_count = loans['status'].isin(['NPL', 'Overdue']).sum()
        kpis['npl_ratio'] = npl_count / total_loans * 100
    else:
        kpis['npl_ratio'] = 0
    
    kpis['total_loans'] = total_loans
    kpis['loan_volume'] = loan_volume
    
    # Распределение кредитов по статусам
    if 'status' in loans.columns:
        kpis['loans_by_status'] = loans['status'].value_counts().to_dict()
    
    # === Депозитные показатели ===
    total_deposits = len(deposits)
    deposit_volume = deposits['principal_amount'].sum() if 'principal_amount' in deposits.columns else 0
    kpis['total_deposits'] = total_deposits
    kpis['deposit_volume'] = deposit_volume
    
    # === LDR (Loan-to-Deposit Ratio) ===
    if deposit_volume > 0:
        kpis['ldr'] = loan_volume / deposit_volume * 100
    else:
        kpis['ldr'] = 0
    
    # === Активы и капитал (оцениваем по данным) ===
    # Активы = кредиты + депозиты клиентов (упрощённо) + остатки на счетах
    total_balance = accounts['balance'].sum() if 'balance' in accounts.columns else 0
    total_assets = loan_volume + total_balance
    
    # Упрощённая оценка капитала (12-15% от активов для типичного банка)
    estimated_capital = total_assets * 0.13
    # Упрощённая оценка прибыли (2% от активов — средняя для региона)
    estimated_profit = total_assets * 0.02
    
    # === ROA = Profit / Assets ===
    if total_assets > 0:
        kpis['roa'] = estimated_profit / total_assets * 100
    else:
        kpis['roa'] = 0
    
    # === ROE = Profit / Capital ===
    if estimated_capital > 0:
        kpis['roe'] = estimated_profit / estimated_capital * 100
    else:
        kpis['roe'] = 0
    
    # === CAR = Capital / Assets ===
    if total_assets > 0:
        kpis['car'] = estimated_capital / total_assets * 100
    else:
        kpis['car'] = 0
    
    # === Клиентские показатели ===
    if 'status' in clients.columns:
        kpis['active_clients'] = (clients['status'] == 'Active').sum()
    else:
        kpis['active_clients'] = len(clients)
    
    kpis['total_clients'] = len(clients)
    
    # === Операционные показатели ===
    if not transactions.empty and 'amount' in transactions.columns:
        kpis['total_transactions'] = len(transactions)
        kpis['transaction_volume'] = transactions['amount'].sum()
        kpis['avg_transaction'] = transactions['amount'].mean()
    
    # === Средний баланс ===
    if 'balance' in accounts.columns and len(accounts) > 0:
        kpis['avg_balance'] = accounts['balance'].mean()
        kpis['total_balance'] = total_balance
    
    return kpis
