import hashlib, json, secrets
from datetime import datetime

key_raw = secrets.token_hex(24)
api_key = 'ne_' + key_raw
key_hash = hashlib.sha256(api_key.encode()).hexdigest()

store_path = 'data/api_keys.json'
with open(store_path, 'r', encoding='utf-8') as f:
    store = json.load(f)

store['keys'][key_hash] = {
    'key_hash': key_hash,
    'key_prefix': api_key[:8] + '...' + api_key[-4:],
    'description': '自动生成',
    'permissions': ['romance'],
    'max_requests': 0,
    'request_count': 0,
    'created_at': datetime.now().isoformat(),
    'expires_at': None,
    'revoked': False,
    'last_used_at': None,
}

with open(store_path, 'w', encoding='utf-8') as f:
    json.dump(store, f, ensure_ascii=False, indent=2)

print(api_key)
