"""
Модуль загрузки реальных файлов с поддержкой Big Data.
Поддерживаемые форматы: CSV, Excel, JSON, Parquet, TSV, Stata (.dta).
Обработка по чанкам для больших файлов.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    '.csv': 'CSV',
    '.tsv': 'TSV',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
    '.json': 'JSON',
    '.parquet': 'Parquet',
    '.dta': 'Stata',
    '.txt': 'Text',
}


def detect_file_type(filename):
    """Определяет тип файла по расширению."""
    ext = Path(filename).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext, 'Unknown')


def get_file_info(file_path):
    """Возвращает информацию о файле (размер, тип, etc.)."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    file_type = detect_file_type(file_path)
    
    return {
        'name': os.path.basename(file_path),
        'size_bytes': size_bytes,
        'size_mb': round(size_mb, 2),
        'type': file_type,
        'is_big': size_mb > 100,  # больше 100 МБ - Big Data режим
    }


def load_file(file_path_or_buffer, file_type=None, filename=None,
              chunksize=None, sample_only=False, sample_size=1000):
    """
    Загружает файл в DataFrame.
    
    Args:
        file_path_or_buffer: путь или буфер файла
        file_type: тип файла (если None — определяется по имени)
        filename: имя файла (для определения типа)
        chunksize: размер чанка для Big Data (None = загружаем всё)
        sample_only: если True — читаем только первые N строк
        sample_size: размер выборки при sample_only=True
    
    Returns:
        DataFrame или генератор чанков
    """
    if file_type is None and filename:
        file_type = detect_file_type(filename)
    
    try:
        # ==== CSV ====
        if file_type in ('CSV', 'TSV', 'Text'):
            sep = '\t' if file_type == 'TSV' else ','
            
            # Пробуем определить разделитель
            if file_type == 'Text':
                # Читаем первые байты чтобы понять разделитель
                try:
                    if hasattr(file_path_or_buffer, 'read'):
                        sample = file_path_or_buffer.read(2048)
                        file_path_or_buffer.seek(0)
                        if isinstance(sample, bytes):
                            sample = sample.decode('utf-8', errors='ignore')
                    else:
                        with open(file_path_or_buffer, 'r', encoding='utf-8', errors='ignore') as f:
                            sample = f.read(2048)
                    
                    if sample.count(';') > sample.count(','):
                        sep = ';'
                    elif sample.count('\t') > sample.count(','):
                        sep = '\t'
                except Exception:
                    sep = ','
            
            kwargs = {
                'sep': sep,
                'encoding': 'utf-8',
                'low_memory': False,
                'on_bad_lines': 'skip',
                # Noto'g'ri int konvertatsiya oldini olish uchun:
                # uzun raqamli ustunlarni string sifatida o'qiymiz
                'dtype': {
                    'account_number': str,
                    'inn': str,
                    'phone': str,
                    'branch_code': str,
                },
            }
            
            if sample_only:
                kwargs['nrows'] = sample_size
                try:
                    return pd.read_csv(file_path_or_buffer, **kwargs)
                except UnicodeDecodeError:
                    kwargs['encoding'] = 'cp1251'
                    return pd.read_csv(file_path_or_buffer, **kwargs)
            
            if chunksize:
                kwargs['chunksize'] = chunksize
                try:
                    return pd.read_csv(file_path_or_buffer, **kwargs)
                except UnicodeDecodeError:
                    kwargs['encoding'] = 'cp1251'
                    return pd.read_csv(file_path_or_buffer, **kwargs)
            
            try:
                return pd.read_csv(file_path_or_buffer, **kwargs)
            except UnicodeDecodeError:
                kwargs['encoding'] = 'cp1251'
                try:
                    return pd.read_csv(file_path_or_buffer, **kwargs)
                except Exception:
                    # Barcha ustunlarni string sifatida o'qiymiz
                    kwargs['dtype'] = str
                    if hasattr(file_path_or_buffer, 'seek'):
                        file_path_or_buffer.seek(0)
                    return pd.read_csv(file_path_or_buffer, **kwargs)
            except (OverflowError, ValueError):
                # Agar int too big yoki boshqa konvertatsiya xatosi - hammasini string qilamiz
                kwargs['dtype'] = str
                if hasattr(file_path_or_buffer, 'seek'):
                    file_path_or_buffer.seek(0)
                return pd.read_csv(file_path_or_buffer, **kwargs)
        
        # ==== Excel ====
        elif file_type == 'Excel':
            kwargs = {}
            if sample_only:
                kwargs['nrows'] = sample_size
            return pd.read_excel(file_path_or_buffer, **kwargs)
        
        # ==== JSON ====
        elif file_type == 'JSON':
            df = pd.read_json(file_path_or_buffer)
            if sample_only:
                df = df.head(sample_size)
            return df
        
        # ==== Parquet ====
        elif file_type == 'Parquet':
            df = pd.read_parquet(file_path_or_buffer)
            if sample_only:
                df = df.head(sample_size)
            return df
        
        # ==== Stata ====
        elif file_type == 'Stata':
            if chunksize:
                return pd.read_stata(file_path_or_buffer, chunksize=chunksize)
            df = pd.read_stata(file_path_or_buffer)
            if sample_only:
                df = df.head(sample_size)
            return df
        
        else:
            raise ValueError(f"Неподдерживаемый формат: {file_type}")
    
    except Exception as e:
        raise Exception(f"Ошибка загрузки файла: {str(e)}")


def load_file_in_chunks(file_path_or_buffer, file_type=None, filename=None,
                         chunksize=100_000, progress_callback=None):
    """
    Загружает большой файл по чанкам с прогрессом.
    Возвращает объединённый DataFrame.
    Для истинной Big Data обработки лучше обрабатывать чанки по отдельности.
    """
    if file_type is None and filename:
        file_type = detect_file_type(filename)
    
    chunks = []
    total_rows = 0
    
    try:
        reader = load_file(file_path_or_buffer, file_type, filename, chunksize=chunksize)
        
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            total_rows += len(chunk)
            if progress_callback:
                progress_callback(i, total_rows)
        
        return pd.concat(chunks, ignore_index=True)
    
    except Exception as e:
        # Если chunks не поддерживаются - грузим целиком
        return load_file(file_path_or_buffer, file_type, filename)


def auto_detect_table_type(df):
    """
    Автоопределение типа банковских данных по колонкам.
    Возвращает: 'clients' / 'accounts' / 'transactions' / 'loans' / 'deposits' / 'unknown'
    """
    cols = set(c.lower() for c in df.columns)
    
    # Клиенты: ИНН, ФИО, дата рождения
    client_indicators = {'inn', 'client_id', 'birth_date', 'birthdate', 'first_name', 'last_name',
                         'фио', 'клиент', 'инн', 'дата_рождения'}
    
    # Счета: номер счёта, баланс
    account_indicators = {'account_number', 'balance', 'account_id', 'account_type',
                          'счет', 'баланс', 'номер_счета'}
    
    # Операции: сумма, дата операции
    transaction_indicators = {'transaction_id', 'transaction_date', 'amount', 'operation_type',
                              'сумма', 'дата_операции', 'операция'}
    
    # Кредиты: процентная ставка, срок, основная сумма
    loan_indicators = {'loan_id', 'interest_rate', 'principal_amount', 'term_months',
                       'maturity_date', 'кредит', 'ставка', 'срок'}
    
    # Депозиты
    deposit_indicators = {'deposit_id', 'maturity_date', 'capitalization',
                          'депозит', 'вклад'}
    
    scores = {
        'clients': len(cols & client_indicators),
        'accounts': len(cols & account_indicators),
        'transactions': len(cols & transaction_indicators),
        'loans': len(cols & loan_indicators),
        'deposits': len(cols & deposit_indicators),
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'unknown'
    
    # Возвращаем тип с максимальным совпадением
    for table_type, score in scores.items():
        if score == max_score:
            return table_type
    
    return 'unknown'


def get_memory_usage(df):
    """Возвращает использование памяти DataFrame в МБ."""
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)


def optimize_dtypes(df):
    """Оптимизирует типы данных для уменьшения памяти (Big Data)."""
    original_mem = get_memory_usage(df)
    
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # object -> category если мало уникальных значений
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_total > 0 and num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    
    new_mem = get_memory_usage(df)
    savings_pct = round((original_mem - new_mem) / original_mem * 100, 1) if original_mem > 0 else 0
    
    return df, {
        'original_mb': original_mem,
        'new_mb': new_mem,
        'savings_mb': round(original_mem - new_mem, 2),
        'savings_pct': savings_pct,
    }
