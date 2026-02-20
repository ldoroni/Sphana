import base64
from typing import Optional

class Base64Util:

    @staticmethod
    def to_base64(plain_value: str | bytes) -> str:
        if isinstance(plain_value, str):
            plain_bytes = plain_value.encode('utf-8')
        else:
            plain_bytes = plain_value
        base64_bytes = base64.b64encode(plain_bytes)
        return base64_bytes.decode('utf-8')

    @staticmethod
    def from_base64(base64_value: str) -> str:
        base64_bytes = base64_value.encode('utf-8')
        input_bytes = base64.b64decode(base64_bytes)
        return input_bytes.decode('utf-8')
    
    @staticmethod
    def to_nullable_base64(plain_value: Optional[str | bytes]) -> Optional[str]:
        if not plain_value:
            return None
        return Base64Util.to_base64(plain_value)
    
    @staticmethod
    def from_nullable_base64(base64_value: Optional[str]) -> Optional[str]:
        if not base64_value:
            return None
        return Base64Util.from_base64(base64_value)