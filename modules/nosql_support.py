"""
NoSQL qo'llab-quvvatlash moduli.
MongoDB, JSON, BSON, hujjatga asoslangan ma'lumotlar bilan ishlash.

NoSQL Module for BigDataPlatform.
Supports: JSON documents, BSON, nested structures, MongoDB-style data.
"""

import pandas as pd
import json
from typing import Dict, List, Any


def load_json_nosql(file_or_path, flatten=True):
    """
    Yuklaydi NoSQL JSON fayllarini.
    Qo'llab-quvvatlaydi: JSON Lines (NDJSON), nested documents, MongoDB exports.
    """
    try:
        # 1-urinish: JSON Lines (har bir qatorda - bitta hujjat)
        if hasattr(file_or_path, 'read'):
            content = file_or_path.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            file_or_path.seek(0)
        else:
            with open(file_or_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Try JSON Lines format
        lines = content.strip().split('\n')
        if len(lines) > 1:
            try:
                docs = [json.loads(line) for line in lines if line.strip()]
                df = pd.json_normalize(docs) if flatten else pd.DataFrame(docs)
                return df, 'jsonlines'
            except json.JSONDecodeError:
                pass
        
        # Try standard JSON
        try:
            data = json.loads(content)
            if isinstance(data, list):
                df = pd.json_normalize(data) if flatten else pd.DataFrame(data)
                return df, 'json_array'
            elif isinstance(data, dict):
                # MongoDB export formati: {"data": [...]}
                for key in ['data', 'documents', 'records', 'items', 'results']:
                    if key in data and isinstance(data[key], list):
                        df = pd.json_normalize(data[key]) if flatten else pd.DataFrame(data[key])
                        return df, f'json_object_{key}'
                
                # Yagona hujjat
                df = pd.json_normalize([data]) if flatten else pd.DataFrame([data])
                return df, 'json_single_doc'
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parse xatosi: {str(e)}")
    
    except Exception as e:
        raise Exception(f"NoSQL yuklash xatosi: {str(e)}")


def load_bson(file_or_path):
    """
    BSON (Binary JSON) fayllarini yuklaydi - MongoDB native format.
    Talab qiladi: pip install pymongo
    """
    try:
        try:
            from bson import decode_all
        except ImportError:
            raise Exception("BSON kutubxonasi o'rnatilmagan. O'rnating: pip install pymongo")
        
        if hasattr(file_or_path, 'read'):
            data = file_or_path.read()
        else:
            with open(file_or_path, 'rb') as f:
                data = f.read()
        
        documents = decode_all(data)
        df = pd.json_normalize(documents)
        return df, 'bson'
    
    except Exception as e:
        raise Exception(f"BSON yuklash xatosi: {str(e)}")


def flatten_nested_json(df, separator='_'):
    """
    Yoyadi ichki (nested) JSON tuzilmalarini.
    Masalan: {"client": {"name": "Ali"}} -> client_name
    """
    while True:
        # Obyekt turidagi ustunlarni topamiz
        object_cols = df.select_dtypes(include=['object']).columns
        has_dicts = False
        
        for col in object_cols:
            # Tekshiramiz - dict bor-yo'qligini
            sample = df[col].dropna().head(5)
            if any(isinstance(v, dict) for v in sample):
                has_dicts = True
                # Yoyamiz
                normalized = pd.json_normalize(df[col].fillna({}))
                normalized.columns = [f"{col}{separator}{c}" for c in normalized.columns]
                df = pd.concat([df.drop(col, axis=1), normalized], axis=1)
        
        if not has_dicts:
            break
    
    return df


def nosql_to_tabular(df):
    """
    NoSQL ma'lumotlarini tabular formatga o'tkazadi va tozalaydi.
    Array ustunlarini alohida qatorlarga ajratadi (explode).
    """
    result = df.copy()
    
    # Array ustunlarni topamiz
    list_cols = []
    for col in result.columns:
        sample = result[col].dropna().head(5)
        if any(isinstance(v, list) for v in sample):
            list_cols.append(col)
    
    # Explode arrays
    for col in list_cols:
        try:
            result = result.explode(col).reset_index(drop=True)
        except Exception:
            # Agar xato bo'lsa, string ga aylantiramiz
            result[col] = result[col].astype(str)
    
    return result


def detect_nosql_structure(df):
    """Aniqlaydi NoSQL ma'lumotlarining tuzilmasini."""
    info = {
        'total_columns': len(df.columns),
        'nested_columns': 0,
        'array_columns': 0,
        'has_mongodb_id': False,
        'depth': 0,
    }
    
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if any(isinstance(v, dict) for v in sample):
            info['nested_columns'] += 1
        elif any(isinstance(v, list) for v in sample):
            info['array_columns'] += 1
        
        if col in ('_id', '$oid', 'ObjectId') or col.startswith('_id'):
            info['has_mongodb_id'] = True
    
    # Chuqurlik - necha daraja ichki
    max_depth = 0
    for col in df.columns:
        depth = col.count('.') + col.count('_') if '.' in col or '_' in col else 0
        max_depth = max(max_depth, depth)
    info['depth'] = max_depth
    
    return info


def get_nosql_sample_data():
    """Demo uchun namuna NoSQL ma'lumotlari."""
    return [
        {
            "_id": "64f1a2b3c4d5e6f7a8b9c0d1",
            "client_id": "CL000001",
            "first_name": "Alisher",
            "last_name": "Karimov",
            "contacts": {
                "phone": "+998901234567",
                "email": "alisher@example.uz",
                "address": {
                    "city": "Toshkent",
                    "region": "Toshkent viloyati"
                }
            },
            "accounts": [
                {"type": "savings", "balance": 5000000, "currency": "UZS"},
                {"type": "current", "balance": 1000, "currency": "USD"}
            ],
            "status": "Active",
            "created_at": "2024-01-15T10:30:00Z"
        },
        {
            "_id": "64f1a2b3c4d5e6f7a8b9c0d2",
            "client_id": "CL000002",
            "first_name": "Malika",
            "last_name": "Yusupova",
            "contacts": {
                "phone": "+998912345678",
                "email": "malika@example.uz",
                "address": {
                    "city": "Samarqand",
                    "region": "Samarqand viloyati"
                }
            },
            "accounts": [
                {"type": "deposit", "balance": 20000000, "currency": "UZS"}
            ],
            "status": "Active",
            "created_at": "2024-03-22T14:15:00Z"
        }
    ]
