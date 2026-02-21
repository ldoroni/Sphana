import httpx
import logging
from abc import ABC
from http import HTTPStatus
from time import time
from typing import Optional
from pydantic import BaseModel
from managed_exceptions import ManagedException, InternalErrorException, UpstreamException
from prometheus_client import Counter, Histogram
from .error_response import ErrorResponse

API_EXE_COUNTER = Counter("spn_client_exe_total", "Total number of API requests executed", ["handler"])
API_EXE_DURATION_HISTOGRAM = Histogram("spn_client_exe_duration_seconds", "Duration of API requests in seconds", ["handler"])
API_EXE_ERROR_COUNTER = Counter("spn_client_exe_error_total", "Total number of API requests that resulted in error", ["handler", "status_code"])

class ClientHandler(ABC):

    def __init__(self, host: str, default_timeout: float = 10.0):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__http_client = httpx.Client(timeout=default_timeout)
        self.__host = host

    def invoke(self, api: str, request: dict, timeout: Optional[float] = None, headers: Optional[dict] = None, cookies: Optional[dict] = None) -> dict:
        start_time: float = time()
        API_EXE_COUNTER.labels(handler=self.__class__.__name__).inc()
        url: str = f"{self.__host.rstrip('/')}:{api}"
        try:
            self.__logger.info(f"[EXTERNAL] Full Request: <{request}>")
            
            # Execute HTTP request
            response = self.__http_client.post(
                url, 
                json=request,
                timeout=timeout or httpx.USE_CLIENT_DEFAULT,
                headers=headers,
                cookies=cookies
            )

            # Parse response
            response_data: dict = response.json()
            if response.status_code == 200:
                return response_data or {}
            else:
                # Return error response
                e1: UpstreamException = UpstreamException(
                    http_status=HTTPStatus(response.status_code),
                    message=response_data.get("message", ""),
                    diagnostic_code=response_data.get("diagnostic_code", ""),
                    diagnostic_details=response_data.get("diagnostic_details", {}),
                )
                actual_error_response: ErrorResponse = self.__get_error_response(e1)
                self.__logger.info(f"Full Response: <{e1.status_code} | {actual_error_response}>")
                API_EXE_ERROR_COUNTER.labels(handler=self.__class__.__name__, status_code=e1.status_code).inc()
                raise e1
        except Exception as e2:
            # Return error response
            actual_error_response: ErrorResponse = self.__get_error_response(e2)
            self.__logger.info(f"Full Response: <{HTTPStatus.INTERNAL_SERVER_ERROR} | {actual_error_response}>", exc_info=True)
            API_EXE_ERROR_COUNTER.labels(handler=self.__class__.__name__, status_code=HTTPStatus.INTERNAL_SERVER_ERROR).inc()
            raise InternalErrorException("An excpected error occurred") from e2
        finally:
            duration: float = time() - start_time
            API_EXE_DURATION_HISTOGRAM.labels(handler=self.__class__.__name__).observe(duration)
        
    def __get_error_response(self, exception: Exception) -> ErrorResponse:
        if isinstance(exception, ManagedException):
            managed_exception = exception
        else:
            managed_exception = InternalErrorException(str(exception))
        
        return ErrorResponse(
            diagnostic_code=managed_exception.diagnostic_code,
            diagnostic_details=managed_exception.diagnostic_details,
            message=str(managed_exception)
        )