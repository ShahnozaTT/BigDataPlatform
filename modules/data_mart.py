"""
Модуль построения витрин данных (Data Marts) и экспорта в SQLite
для подключения к Apache Superset / Power BI / Tableau.
"""

import pandas as pd
import numpy as np
import sqlite3


def build_marts(data_dict):
    """Строит агрегированные витрины данных для BI-инструментов."""
    clients = data_dict.get('clients', pd.DataFrame())
    accounts = data_dict.get('accounts', pd.DataFrame())
    transactions = data_dict.get('transactions', pd.DataFrame())
    loans = data_dict.get('loans', pd.DataFrame())
    deposits = data_dict.get('deposits', pd.DataFrame())
    
    marts = {}
    
    # ===== ВИТРИНА 1: DIM_CLIENTS =====
    if not clients.empty:
        marts['dim_clients'] = clients.copy()
    
    # ===== ВИТРИНА 2: DIM_ACCOUNTS =====
    if not accounts.empty:
        marts['dim_accounts'] = accounts.copy()
    
    # ===== ВИТРИНА 3: FACT_TRANSACTIONS =====
    if not transactions.empty:
        fact = transactions.copy()
        if 'transaction_date' in fact.columns:
            fact['transaction_date'] = pd.to_datetime(fact['transaction_date'], errors='coerce')
            fact['year'] = fact['transaction_date'].dt.year
            fact['month'] = fact['transaction_date'].dt.month
            fact['day'] = fact['transaction_date'].dt.day
            fact['weekday'] = fact['transaction_date'].dt.day_name()
            fact['quarter'] = fact['transaction_date'].dt.quarter
        marts['fact_transactions'] = fact
    
    # ===== ВИТРИНА 4: FACT_LOANS =====
    if not loans.empty:
        fact = loans.copy()
        if 'issue_date' in fact.columns:
            fact['issue_date'] = pd.to_datetime(fact['issue_date'], errors='coerce')
            fact['issue_year'] = fact['issue_date'].dt.year
            fact['issue_month'] = fact['issue_date'].dt.month
            fact['issue_quarter'] = fact['issue_date'].dt.quarter
        # флаг проблемного кредита
        if 'status' in fact.columns:
            fact['is_npl'] = fact['status'].isin(['NPL', 'Overdue']).astype(int)
        marts['fact_loans'] = fact
    
    # ===== ВИТРИНА 5: FACT_DEPOSITS =====
    if not deposits.empty:
        fact = deposits.copy()
        if 'open_date' in fact.columns:
            fact['open_date'] = pd.to_datetime(fact['open_date'], errors='coerce')
            fact['open_year'] = fact['open_date'].dt.year
            fact['open_month'] = fact['open_date'].dt.month
            fact['open_quarter'] = fact['open_date'].dt.quarter
        marts['fact_deposits'] = fact
    
    # ===== ВИТРИНА 6: AGG_MONTHLY_TRANSACTIONS =====
    if not transactions.empty and 'transaction_date' in transactions.columns:
        tx = transactions.copy()
        tx['transaction_date'] = pd.to_datetime(tx['transaction_date'], errors='coerce')
        tx['year_month'] = tx['transaction_date'].dt.to_period('M').astype(str)
        
        agg = tx.groupby(['year_month', 'operation_type']).agg(
            transaction_count=('transaction_id', 'count'),
            total_amount=('amount', 'sum'),
            avg_amount=('amount', 'mean'),
        ).reset_index()
        marts['agg_monthly_transactions'] = agg
    
    # ===== ВИТРИНА 7: AGG_LOANS_BY_REGION =====
    if not loans.empty and not clients.empty:
        merged = loans.merge(
            clients[['client_id', 'region']],
            on='client_id',
            how='left'
        )
        if 'region' in merged.columns:
            agg = merged.groupby('region').agg(
                loan_count=('loan_id', 'count'),
                total_principal=('principal_amount', 'sum'),
                avg_interest_rate=('interest_rate', 'mean'),
                npl_count=('status', lambda x: x.isin(['NPL', 'Overdue']).sum()),
            ).reset_index()
            agg['npl_ratio_pct'] = (agg['npl_count'] / agg['loan_count'] * 100).round(2)
            marts['agg_loans_by_region'] = agg
    
    # ===== ВИТРИНА 8: AGG_CLIENT_PORTFOLIO =====
    if not clients.empty:
        portfolio = clients[['client_id', 'region', 'client_type', 'status']].copy()
        
        # Кредиты по клиентам
        if not loans.empty and 'client_id' in loans.columns:
            loan_agg = loans.groupby('client_id').agg(
                loans_count=('loan_id', 'count'),
                loans_total=('principal_amount', 'sum'),
            ).reset_index()
            portfolio = portfolio.merge(loan_agg, on='client_id', how='left')
            portfolio['loans_count'] = portfolio['loans_count'].fillna(0).astype(int)
            portfolio['loans_total'] = portfolio['loans_total'].fillna(0)
        
        # Депозиты по клиентам
        if not deposits.empty and 'client_id' in deposits.columns:
            dep_agg = deposits.groupby('client_id').agg(
                deposits_count=('deposit_id', 'count'),
                deposits_total=('principal_amount', 'sum'),
            ).reset_index()
            portfolio = portfolio.merge(dep_agg, on='client_id', how='left')
            portfolio['deposits_count'] = portfolio['deposits_count'].fillna(0).astype(int)
            portfolio['deposits_total'] = portfolio['deposits_total'].fillna(0)
        
        marts['agg_client_portfolio'] = portfolio
    
    # ===== ВИТРИНА 9: DIM_DATE (календарь) =====
    if not transactions.empty and 'transaction_date' in transactions.columns:
        tx_dates = pd.to_datetime(transactions['transaction_date'], errors='coerce').dropna()
        if not tx_dates.empty:
            date_range = pd.date_range(
                start=tx_dates.min().floor('D'),
                end=tx_dates.max().ceil('D'),
                freq='D'
            )
            dim_date = pd.DataFrame({'date': date_range})
            dim_date['year'] = dim_date['date'].dt.year
            dim_date['quarter'] = dim_date['date'].dt.quarter
            dim_date['month'] = dim_date['date'].dt.month
            dim_date['month_name'] = dim_date['date'].dt.strftime('%B')
            dim_date['day'] = dim_date['date'].dt.day
            dim_date['weekday'] = dim_date['date'].dt.day_name()
            dim_date['is_weekend'] = dim_date['date'].dt.dayofweek.isin([5, 6]).astype(int)
            marts['dim_date'] = dim_date
    
    return marts


def export_to_sqlite(marts, db_path):
    """Экспортирует витрины в SQLite для подключения к BI-инструментам."""
    conn = sqlite3.connect(db_path)
    try:
        for mart_name, mart_df in marts.items():
            # конвертируем datetime в строку для совместимости
            df_to_write = mart_df.copy()
            for col in df_to_write.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_write[col]):
                    df_to_write[col] = df_to_write[col].astype(str)
            df_to_write.to_sql(mart_name, conn, if_exists='replace', index=False)
        conn.commit()
    finally:
        conn.close()
    return db_path
