"""
Модуль расширенной банковской аналитики.
Анализы: отток клиентов, когорты, концентрация рисков, ABC-анализ,
валютный анализ, анализ ликвидности.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def analyze_currency(accounts_df, transactions_df=None):
    """Валютный анализ: распределение по валютам."""
    if accounts_df.empty or 'currency' not in accounts_df.columns:
        return None
    
    result = {}
    
    if 'balance' in accounts_df.columns:
        currency_balance = accounts_df.groupby('currency').agg(
            total_balance=('balance', 'sum'),
            avg_balance=('balance', 'mean'),
            account_count=('balance', 'count'),
        ).reset_index()
        currency_balance['share_pct'] = (
            currency_balance['total_balance'] / currency_balance['total_balance'].sum() * 100
        ).round(2)
        result['by_accounts'] = currency_balance
    
    if transactions_df is not None and not transactions_df.empty and 'currency' in transactions_df.columns:
        currency_tx = transactions_df.groupby('currency').agg(
            total_amount=('amount', 'sum'),
            tx_count=('amount', 'count'),
            avg_amount=('amount', 'mean'),
        ).reset_index()
        result['by_transactions'] = currency_tx
    
    return result


def analyze_churn(clients_df, transactions_df=None, threshold_days=90):
    """
    Анализ оттока клиентов (churn analysis).
    Клиент считается "ушедшим", если не совершал операций более threshold_days.
    """
    if clients_df.empty:
        return None
    
    result = {
        'total_clients': len(clients_df),
    }
    
    if 'status' in clients_df.columns:
        status_dist = clients_df['status'].value_counts().to_dict()
        result['by_status'] = status_dist
        
        active = status_dist.get('Active', 0)
        inactive = status_dist.get('Inactive', 0) + status_dist.get('Blocked', 0)
        total = active + inactive
        result['churn_rate_pct'] = round(inactive / total * 100, 2) if total > 0 else 0
    
    if transactions_df is not None and not transactions_df.empty:
        if 'account_id' in transactions_df.columns and 'transaction_date' in transactions_df.columns:
            tx = transactions_df.copy()
            tx['transaction_date'] = pd.to_datetime(tx['transaction_date'], errors='coerce')
            last_tx = tx.groupby('account_id')['transaction_date'].max().reset_index()
            last_tx['days_since'] = (pd.Timestamp.now() - last_tx['transaction_date']).dt.days
            
            result['dormant_accounts'] = int((last_tx['days_since'] > threshold_days).sum())
            result['active_accounts'] = int((last_tx['days_since'] <= threshold_days).sum())
    
    return result


def abc_analysis(clients_df, loans_df=None, deposits_df=None):
    """
    ABC-анализ клиентов по объёму портфеля.
    A: топ-20% клиентов, дающих 80% объёма
    B: следующие 30% клиентов
    C: оставшиеся 50%
    """
    if clients_df.empty:
        return None
    
    # Построим портфель на клиента
    portfolio = clients_df[['client_id']].copy()
    portfolio['total_volume'] = 0
    
    if loans_df is not None and not loans_df.empty and 'client_id' in loans_df.columns:
        loan_sum = loans_df.groupby('client_id')['principal_amount'].sum().reset_index()
        loan_sum.columns = ['client_id', 'loan_volume']
        portfolio = portfolio.merge(loan_sum, on='client_id', how='left')
        portfolio['loan_volume'] = portfolio['loan_volume'].fillna(0)
        portfolio['total_volume'] += portfolio['loan_volume']
    
    if deposits_df is not None and not deposits_df.empty and 'client_id' in deposits_df.columns:
        dep_sum = deposits_df.groupby('client_id')['principal_amount'].sum().reset_index()
        dep_sum.columns = ['client_id', 'deposit_volume']
        portfolio = portfolio.merge(dep_sum, on='client_id', how='left')
        portfolio['deposit_volume'] = portfolio['deposit_volume'].fillna(0)
        portfolio['total_volume'] += portfolio['deposit_volume']
    
    # Сортируем по объёму
    portfolio = portfolio.sort_values('total_volume', ascending=False).reset_index(drop=True)
    
    # Присваиваем группы ABC
    n = len(portfolio)
    if n == 0:
        return None
    
    portfolio['abc_group'] = 'C'
    portfolio.loc[:int(n * 0.2), 'abc_group'] = 'A'
    portfolio.loc[int(n * 0.2):int(n * 0.5), 'abc_group'] = 'B'
    
    summary = portfolio.groupby('abc_group').agg(
        client_count=('client_id', 'count'),
        total_volume=('total_volume', 'sum'),
        avg_volume=('total_volume', 'mean'),
    ).reset_index()
    summary['share_pct'] = (summary['total_volume'] / summary['total_volume'].sum() * 100).round(2)
    
    return {
        'summary': summary,
        'portfolio': portfolio,
    }


def analyze_concentration_risk(loans_df):
    """
    Анализ концентрации кредитного риска.
    Что происходит, если крупнейшие заёмщики дефолтят.
    """
    if loans_df.empty or 'principal_amount' not in loans_df.columns:
        return None
    
    total = loans_df['principal_amount'].sum()
    sorted_loans = loans_df.sort_values('principal_amount', ascending=False)
    
    result = {
        'total_portfolio': round(total, 2),
        'loan_count': len(loans_df),
    }
    
    # Top 10, 20, 50, 100 крупнейших кредитов
    for top_n in [10, 20, 50, 100]:
        if len(sorted_loans) >= top_n:
            top_sum = sorted_loans.head(top_n)['principal_amount'].sum()
            result[f'top_{top_n}_share_pct'] = round(top_sum / total * 100, 2)
            result[f'top_{top_n}_volume'] = round(top_sum, 2)
    
    # По продуктам
    if 'product_type' in loans_df.columns:
        by_product = loans_df.groupby('product_type').agg(
            count=('principal_amount', 'count'),
            total=('principal_amount', 'sum'),
        ).reset_index()
        by_product['share_pct'] = (by_product['total'] / total * 100).round(2)
        result['by_product'] = by_product
    
    return result


def analyze_liquidity(loans_df, deposits_df):
    """
    Анализ ликвидности: соответствие сроков кредитов и депозитов.
    """
    result = {}
    
    if not loans_df.empty and 'term_months' in loans_df.columns:
        loan_buckets = pd.cut(
            loans_df['term_months'],
            bins=[0, 12, 36, 60, 120, 999],
            labels=['<1 год', '1-3 года', '3-5 лет', '5-10 лет', '>10 лет']
        )
        loan_by_term = loans_df.groupby(loan_buckets, observed=True)['principal_amount'].agg(
            ['sum', 'count']
        ).reset_index()
        loan_by_term.columns = ['term_bucket', 'loan_volume', 'loan_count']
        result['loans_by_term'] = loan_by_term
    
    if not deposits_df.empty and 'term_months' in deposits_df.columns:
        dep_buckets = pd.cut(
            deposits_df['term_months'],
            bins=[0, 3, 12, 36, 999],
            labels=['<3 мес', '3-12 мес', '1-3 года', '>3 лет']
        )
        dep_by_term = deposits_df.groupby(dep_buckets, observed=True)['principal_amount'].agg(
            ['sum', 'count']
        ).reset_index()
        dep_by_term.columns = ['term_bucket', 'deposit_volume', 'deposit_count']
        result['deposits_by_term'] = dep_by_term
    
    # Общий gap-анализ
    total_loans = loans_df['principal_amount'].sum() if not loans_df.empty else 0
    total_deposits = deposits_df['principal_amount'].sum() if not deposits_df.empty else 0
    
    result['total_loans'] = round(total_loans, 2)
    result['total_deposits'] = round(total_deposits, 2)
    result['liquidity_gap'] = round(total_deposits - total_loans, 2)
    result['ldr_ratio'] = round(total_loans / total_deposits * 100, 2) if total_deposits > 0 else 0
    
    return result


def channel_analysis(transactions_df):
    """Анализ каналов обслуживания: Branch/Mobile/Web/ATM/POS."""
    if transactions_df.empty or 'channel' not in transactions_df.columns:
        return None
    
    result = transactions_df.groupby('channel').agg(
        tx_count=('amount', 'count'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
    ).reset_index()
    
    result['count_share_pct'] = (result['tx_count'] / result['tx_count'].sum() * 100).round(2)
    result['volume_share_pct'] = (result['total_amount'] / result['total_amount'].sum() * 100).round(2)
    
    return result


def regional_analysis(clients_df, loans_df=None, deposits_df=None):
    """Региональный анализ по областям Узбекистана."""
    if clients_df.empty or 'region' not in clients_df.columns:
        return None
    
    result = clients_df.groupby('region').agg(
        clients_count=('client_id', 'count'),
    ).reset_index()
    
    if loans_df is not None and not loans_df.empty and 'client_id' in loans_df.columns:
        merged = loans_df.merge(clients_df[['client_id', 'region']], on='client_id', how='left')
        loan_reg = merged.groupby('region').agg(
            loan_count=('loan_id', 'count'),
            loan_volume=('principal_amount', 'sum'),
        ).reset_index()
        result = result.merge(loan_reg, on='region', how='left')
    
    if deposits_df is not None and not deposits_df.empty and 'client_id' in deposits_df.columns:
        merged = deposits_df.merge(clients_df[['client_id', 'region']], on='client_id', how='left')
        dep_reg = merged.groupby('region').agg(
            deposit_count=('deposit_id', 'count'),
            deposit_volume=('principal_amount', 'sum'),
        ).reset_index()
        result = result.merge(dep_reg, on='region', how='left')
    
    return result.fillna(0)


def temporal_analysis(transactions_df):
    """Временной анализ операций: тренды по месяцам, дням недели."""
    if transactions_df.empty or 'transaction_date' not in transactions_df.columns:
        return None
    
    tx = transactions_df.copy()
    tx['transaction_date'] = pd.to_datetime(tx['transaction_date'], errors='coerce')
    tx = tx.dropna(subset=['transaction_date'])
    
    result = {}
    
    # По месяцам
    tx['year_month'] = tx['transaction_date'].dt.to_period('M').astype(str)
    monthly = tx.groupby('year_month').agg(
        tx_count=('amount', 'count'),
        total_amount=('amount', 'sum'),
    ).reset_index()
    result['monthly'] = monthly
    
    # По дням недели
    tx['weekday'] = tx['transaction_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday = tx.groupby('weekday').agg(
        tx_count=('amount', 'count'),
        total_amount=('amount', 'sum'),
    ).reindex(weekday_order).reset_index()
    result['weekday'] = weekday
    
    # По часам (если есть время)
    try:
        tx['hour'] = tx['transaction_date'].dt.hour
        hourly = tx.groupby('hour').agg(
            tx_count=('amount', 'count'),
            total_amount=('amount', 'sum'),
        ).reset_index()
        result['hourly'] = hourly
    except Exception:
        pass
    
    return result


def comprehensive_analysis(data_dict):
    """Выполняет все виды анализа сразу."""
    clients = data_dict.get('clients', pd.DataFrame())
    accounts = data_dict.get('accounts', pd.DataFrame())
    transactions = data_dict.get('transactions', pd.DataFrame())
    loans = data_dict.get('loans', pd.DataFrame())
    deposits = data_dict.get('deposits', pd.DataFrame())
    
    return {
        'currency': analyze_currency(accounts, transactions),
        'churn': analyze_churn(clients, transactions),
        'abc': abc_analysis(clients, loans, deposits),
        'concentration': analyze_concentration_risk(loans),
        'liquidity': analyze_liquidity(loans, deposits),
        'channels': channel_analysis(transactions),
        'regional': regional_analysis(clients, loans, deposits),
        'temporal': temporal_analysis(transactions),
    }
