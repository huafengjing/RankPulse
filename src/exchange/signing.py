from __future__ import annotations

import hmac
from hashlib import sha256


def sign_query(query_string: str, api_secret: str) -> str:
    return hmac.new(api_secret.encode("utf-8"), query_string.encode("utf-8"), sha256).hexdigest()
