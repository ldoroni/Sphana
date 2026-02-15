import base64
from typing import Optional

class Base64Util:

    @staticmethod
    def to_base64(input_str: str) -> str:
        input_bytes = input_str.encode('utf-8')
        base64_bytes = base64.urlsafe_b64encode(input_bytes)
        return base64_bytes.decode('utf-8')

    @staticmethod
    def from_base64(base64_str: str) -> str:
        base64_bytes = base64_str.encode('utf-8')
        input_bytes = base64.urlsafe_b64decode(base64_bytes)
        return input_bytes.decode('utf-8')
    
    @staticmethod
    def to_nullable_base64(plain_string: Optional[str]) -> Optional[str]:
        if not plain_string:
            return None
        return Base64Util.to_base64(plain_string)
    
    @staticmethod
    def from_nullable_base64(base64_string: Optional[str]) -> Optional[str]:
        if not base64_string:
            return None
        return Base64Util.from_base64(base64_string)