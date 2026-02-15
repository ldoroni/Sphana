from typing import Optional
import zlib

class CompressionUtil:

    @staticmethod
    def compress(plain_data: str) -> bytes:
        byte_data: bytes = plain_data.encode('utf-8')
        return zlib.compress(byte_data)

    @staticmethod
    def decompress(compressed_data: bytes) -> str:
        decompressed_bytes: bytes = zlib.decompress(compressed_data)
        return decompressed_bytes.decode('utf-8')

    @staticmethod
    def compress_nullable(plain_data: Optional[str]) -> Optional[bytes]:
        if not plain_data:
            return None
        return CompressionUtil.compress(plain_data)

    @staticmethod
    def decompress_nullable(compressed_data: Optional[bytes]) -> Optional[str]:
        if not compressed_data:
            return None
        return CompressionUtil.decompress(compressed_data)