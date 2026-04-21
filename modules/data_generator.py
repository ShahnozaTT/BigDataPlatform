"""
Модуль генерации синтетических банковских данных.
Создаёт реалистичные данные с намеренно внесёнными 'грязными' записями
для демонстрации возможностей очистки.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker('ru_RU')
Faker.seed(42)
np.random.seed(42)
random.seed(42)


def generate_clients(n=1000, dirty_pct=0.15):
    """Генерирует таблицу клиентов."""
    clients = []
    for i in range(n):
        gender = random.choice(['M', 'F'])
        first_name = fake.first_name_male() if gender == 'M' else fake.first_name_female()
        last_name = fake.last_name_male() if gender == 'M' else fake.last_name_female()
        
        clients.append({
            'client_id': f'CL{i+1:06d}',
            'first_name': first_name,
            'last_name': last_name,
            'gender': gender,
            'birth_date': fake.date_of_birth(minimum_age=18, maximum_age=85),
            'inn': fake.numerify('#########'),
            'phone': fake.phone_number(),
            'region': random.choice([
                'Toshkent shahri', 'Toshkent viloyati', 'Samarqand', 'Buxoro',
                'Andijon', 'Farg\'ona', 'Namangan', 'Qashqadaryo', 'Surxondaryo',
                'Jizzax', 'Sirdaryo', 'Navoiy', 'Xorazm', 'Qoraqalpog\'iston'
            ]),
            'client_type': random.choice(['Individual', 'Individual', 'Individual', 'Corporate']),
            'registration_date': fake.date_between(start_date='-10y', end_date='today'),
            'status': random.choice(['Active', 'Active', 'Active', 'Inactive', 'Blocked']),
        })
    
    df = pd.DataFrame(clients)
    df = _introduce_dirt(df, dirty_pct, 
                         nullable=['phone', 'region', 'inn'],
                         string_cols=['first_name', 'last_name'])
    return df


def generate_accounts(clients_df, n=1500, dirty_pct=0.15):
    """Генерирует таблицу счетов."""
    accounts = []
    client_ids = clients_df['client_id'].dropna().tolist()
    
    for i in range(n):
        account_type = random.choice(['CURRENT', 'SAVINGS', 'DEPOSIT', 'LOAN', 'CARD'])
        accounts.append({
            'account_id': f'AC{i+1:08d}',
            'account_number': fake.numerify('####################'),  # 20 digits
            'client_id': random.choice(client_ids),
            'account_type': account_type,
            'currency': random.choices(['UZS', 'USD', 'EUR', 'RUB'], 
                                        weights=[70, 20, 7, 3])[0],
            'balance': round(random.uniform(0, 500_000_000), 2),
            'open_date': fake.date_between(start_date='-5y', end_date='today'),
            'status': random.choice(['Active', 'Active', 'Active', 'Closed', 'Frozen']),
            'branch_code': f'BR{random.randint(1, 200):03d}',
        })
    
    df = pd.DataFrame(accounts)
    df = _introduce_dirt(df, dirty_pct,
                         nullable=['branch_code', 'currency'],
                         numeric_cols=['balance'])
    return df


def generate_transactions(accounts_df, n=10000, dirty_pct=0.15):
    """Генерирует таблицу операций."""
    transactions = []
    account_ids = accounts_df['account_id'].dropna().tolist()
    
    for i in range(n):
        op_type = random.choice(['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT', 'FEE'])
        amount = round(random.uniform(1000, 50_000_000), 2)
        
        transactions.append({
            'transaction_id': f'TX{i+1:010d}',
            'account_id': random.choice(account_ids),
            'transaction_date': fake.date_time_between(start_date='-2y', end_date='now'),
            'operation_type': op_type,
            'amount': amount,
            'currency': random.choice(['UZS', 'UZS', 'UZS', 'USD']),
            'description': fake.sentence(nb_words=5),
            'status': random.choice(['Completed', 'Completed', 'Completed', 'Pending', 'Failed']),
            'channel': random.choice(['Branch', 'Mobile', 'Web', 'ATM', 'POS']),
        })
    
    df = pd.DataFrame(transactions)
    df = _introduce_dirt(df, dirty_pct,
                         nullable=['description', 'channel'],
                         numeric_cols=['amount'])
    return df


def generate_loans(clients_df, n=500, dirty_pct=0.15):
    """Генерирует таблицу кредитов."""
    loans = []
    client_ids = clients_df['client_id'].dropna().tolist()
    
    for i in range(n):
        principal = round(random.uniform(1_000_000, 500_000_000), 2)
        term = random.choice([6, 12, 24, 36, 48, 60, 120])
        rate = round(random.uniform(14, 32), 2)
        issue_date = fake.date_between(start_date='-3y', end_date='today')
        
        # Статус кредита
        status_choice = random.choices(
            ['Active', 'Closed', 'Overdue', 'NPL', 'Restructured'],
            weights=[60, 20, 10, 7, 3]
        )[0]
        
        loans.append({
            'loan_id': f'LN{i+1:08d}',
            'client_id': random.choice(client_ids),
            'product_type': random.choice([
                'Consumer', 'Mortgage', 'Auto', 'Business', 'MicroLoan', 'CreditCard'
            ]),
            'principal_amount': principal,
            'interest_rate': rate,
            'term_months': term,
            'issue_date': issue_date,
            'maturity_date': issue_date + timedelta(days=term * 30),
            'outstanding_balance': round(principal * random.uniform(0, 1), 2) if status_choice != 'Closed' else 0,
            'status': status_choice,
            'days_overdue': random.randint(0, 180) if status_choice in ['Overdue', 'NPL'] else 0,
            'currency': random.choices(['UZS', 'USD'], weights=[85, 15])[0],
        })
    
    df = pd.DataFrame(loans)
    df = _introduce_dirt(df, dirty_pct,
                         nullable=['product_type'],
                         numeric_cols=['interest_rate', 'principal_amount'])
    return df


def generate_deposits(clients_df, n=800, dirty_pct=0.15):
    """Генерирует таблицу депозитов."""
    deposits = []
    client_ids = clients_df['client_id'].dropna().tolist()
    
    for i in range(n):
        amount = round(random.uniform(500_000, 1_000_000_000), 2)
        term = random.choice([3, 6, 12, 24, 36])
        rate = round(random.uniform(14, 24), 2)
        open_date = fake.date_between(start_date='-3y', end_date='today')
        
        deposits.append({
            'deposit_id': f'DP{i+1:08d}',
            'client_id': random.choice(client_ids),
            'product_type': random.choice([
                'TimeDeposit', 'SavingsAccount', 'CertificateOfDeposit', 'RetirementDeposit'
            ]),
            'principal_amount': amount,
            'interest_rate': rate,
            'term_months': term,
            'open_date': open_date,
            'maturity_date': open_date + timedelta(days=term * 30),
            'current_balance': round(amount * random.uniform(1, 1.25), 2),
            'status': random.choices(['Active', 'Matured', 'Withdrawn'], weights=[70, 20, 10])[0],
            'currency': random.choices(['UZS', 'USD'], weights=[80, 20])[0],
            'capitalization': random.choice(['Monthly', 'Quarterly', 'AtMaturity']),
        })
    
    df = pd.DataFrame(deposits)
    df = _introduce_dirt(df, dirty_pct,
                         nullable=['capitalization'],
                         numeric_cols=['interest_rate'])
    return df


def _introduce_dirt(df, dirty_pct, nullable=None, string_cols=None, numeric_cols=None):
    """Вносит 'грязные' данные: пропуски, дубликаты, аномалии."""
    nullable = nullable or []
    string_cols = string_cols or []
    numeric_cols = numeric_cols or []
    
    n_dirty = int(len(df) * dirty_pct)
    if n_dirty == 0:
        return df
    
    # 1. Добавляем NULL значения в nullable колонки
    for col in nullable:
        if col in df.columns:
            null_idx = np.random.choice(df.index, size=n_dirty // 2, replace=False)
            df.loc[null_idx, col] = None
    
    # 2. Вносим проблемы в строковые поля (пробелы, регистр)
    for col in string_cols:
        if col in df.columns:
            dirty_idx = np.random.choice(df.index, size=n_dirty // 4, replace=False)
            for idx in dirty_idx:
                if pd.notna(df.loc[idx, col]):
                    val = str(df.loc[idx, col])
                    # случайные пробелы/регистр
                    df.loc[idx, col] = random.choice([
                        f"  {val}  ",
                        val.upper(),
                        val.lower(),
                        f" {val}",
                    ])
    
    # 3. Добавляем аномалии в числовые поля
    for col in numeric_cols:
        if col in df.columns:
            anomaly_idx = np.random.choice(df.index, size=n_dirty // 5, replace=False)
            for idx in anomaly_idx:
                df.loc[idx, col] = random.choice([
                    -abs(df.loc[idx, col]),  # отрицательное значение
                    df.loc[idx, col] * 1000,  # выброс
                    None,  # пропуск
                ])
    
    # 4. Добавляем полные дубликаты
    n_dupes = n_dirty // 3
    if n_dupes > 0 and len(df) > 0:
        dupe_rows = df.sample(n=min(n_dupes, len(df)))
        df = pd.concat([df, dupe_rows], ignore_index=True)
    
    return df


def generate_all_data(n_clients=1000, n_accounts=1500, n_transactions=10000,
                       n_loans=500, n_deposits=800, dirty_pct=0.15):
    """Генерирует все таблицы сразу."""
    clients = generate_clients(n_clients, dirty_pct)
    accounts = generate_accounts(clients, n_accounts, dirty_pct)
    transactions = generate_transactions(accounts, n_transactions, dirty_pct)
    loans = generate_loans(clients, n_loans, dirty_pct)
    deposits = generate_deposits(clients, n_deposits, dirty_pct)
    
    return {
        'clients': clients,
        'accounts': accounts,
        'transactions': transactions,
        'loans': loans,
        'deposits': deposits,
    }
