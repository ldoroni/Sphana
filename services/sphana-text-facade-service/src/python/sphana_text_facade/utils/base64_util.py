import base64
from typing import Optional

class Base64Util:

    @staticmethod
    def encode_to_bytes(plain_value: str | bytes) -> bytes:
        if isinstance(plain_value, str):
            plain_bytes = plain_value.encode('utf-8')
        else:
            plain_bytes = plain_value
        return base64.b64encode(plain_bytes)

    @staticmethod
    def encode_to_str(plain_value: str | bytes) -> str:
        base64_bytes = Base64Util.encode_to_bytes(plain_value)
        return base64_bytes.decode('utf-8')

    @staticmethod
    def decode_to_bytes(base64_value: str | bytes) -> bytes:
        if isinstance(base64_value, str):
            base64_bytes = base64_value.encode('utf-8')
        else:
            base64_bytes = base64_value
        return base64.b64decode(base64_bytes)
    
    @staticmethod
    def decode_to_str(base64_value: str | bytes) -> str:
        plain_bytes = Base64Util.decode_to_bytes(base64_value)
        return plain_bytes.decode('utf-8')
    
    @staticmethod
    def encode_nullable_to_bytes(plain_value: Optional[str | bytes]) -> Optional[bytes]:
        if not plain_value:
            return None
        return Base64Util.encode_to_bytes(plain_value)
    
    @staticmethod
    def encode_nullable_to_str(plain_value: Optional[str | bytes]) -> Optional[str]:
        if not plain_value:
            return None
        return Base64Util.encode_to_str(plain_value)
    
    @staticmethod
    def decode_nullable_to_bytes(base64_value: Optional[str | bytes]) -> Optional[bytes]:
        if not base64_value:
            return None
        return Base64Util.decode_to_bytes(base64_value)
    
    @staticmethod
    def decode_nullable_to_str(base64_value: Optional[str | bytes]) -> Optional[str]:
        if not base64_value:
            return None
        return Base64Util.decode_to_str(base64_value)
