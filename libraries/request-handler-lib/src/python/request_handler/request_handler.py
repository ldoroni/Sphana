import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from time import time
from typing import Any, get_args
from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from managed_exceptions import ManagedException, InternalErrorException
from pydantic import TypeAdapter
from prometheus_client import Counter, Histogram, Gauge
from request_handler.error_response import ErrorResponse
from .request_thread_pool import RequestThreadPool

CONCURRENT_REQUESTS = Gauge("spn_api_exe_concurrent", "Number of concurrent API requests being processed", ["handler"])
REQUEST_COUNTER = Counter("spn_api_exe_total", "Total number of API requests executed", ["handler"])
REQUEST_HISTOGRAM = Histogram("spn_api_exe_duration_seconds", "Duration of API requests in seconds", ["handler"])
RESPONSE_ERROR_COUNTER = Counter("spn_api_exe_error_total", "Total number of API requests that resulted in error", ["handler", "status_code"])

class RequestHandler[TRequest, TResponse](ABC):

    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)
        base_type = self.__class__.__orig_bases__[0] # type: ignore
        generics = get_args(base_type)
        request_class = generics[0]
        self.__request_type_adapter = TypeAdapter(request_class)

    async def invoke(self, request: Request) -> Response:
        start_time: float = time()
        REQUEST_COUNTER.labels(handler=self.__class__.__name__).inc()
        CONCURRENT_REQUESTS.labels(handler=self.__class__.__name__).inc()
        try:
            # Get request body
            body = await request.body()
            actual_request: TRequest = self.__request_type_adapter.validate_json(body)
            # Invoke request
            self.__logger.info(f"Full Request: <{actual_request}>")
            actual_ok_response: TResponse = await RequestThreadPool.run_async(lambda: self.__invoke_sync(actual_request))
            # Return OK response
            self.__logger.info(f"Full Response: <{HTTPStatus.OK} | {actual_ok_response}>")
            return self.__get_response(HTTPStatus.OK, actual_ok_response)
        except ManagedException as e:
            # Return error response
            actual_error_response: ErrorResponse = self.__get_error_response(e)
            self.__logger.info(f"Full Response: <{e.status_code} | {actual_error_response}>")
            RESPONSE_ERROR_COUNTER.labels(handler=self.__class__.__name__, status_code=e.status_code).inc()
            return self.__get_response(e.status_code, actual_error_response)
        except Exception as e:
            # Return error response
            actual_error_response: ErrorResponse = self.__get_error_response(e)
            self.__logger.info(f"Full Response: <{HTTPStatus.INTERNAL_SERVER_ERROR} | {actual_error_response}>", exc_info=True)
            RESPONSE_ERROR_COUNTER.labels(handler=self.__class__.__name__, status_code=HTTPStatus.INTERNAL_SERVER_ERROR).inc()
            return self.__get_response(HTTPStatus.INTERNAL_SERVER_ERROR, actual_error_response)
        finally:
            duration: float = time() - start_time
            REQUEST_HISTOGRAM.labels(handler=self.__class__.__name__).observe(duration)
            CONCURRENT_REQUESTS.labels(handler=self.__class__.__name__).dec()
    
    def __invoke_sync(self, request: TRequest) -> TResponse:
        # Validate request
        self._on_validate(request)
        # Invoke handler
        return self._on_invoke(request);

    @abstractmethod
    def _on_validate(self, request: TRequest) -> None:
        pass

    @abstractmethod
    def _on_invoke(self, request: TRequest) -> TResponse:
        pass

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

    def __get_response(self, status_code: HTTPStatus, content: Any) -> JSONResponse:
        return JSONResponse(
            status_code=status_code, 
            content=jsonable_encoder(content)
        )
    