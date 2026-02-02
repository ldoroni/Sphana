import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, get_args
from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from managed_exceptions import ManagedException, InternalErrorException
from pydantic import TypeAdapter
from request_handler.error_response import ErrorResponse

class RequestHandler[TRequest, TResponse](ABC):
    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)
        base_type = self.__class__.__orig_bases__[0] # type: ignore
        generics = get_args(base_type)
        request_class = generics[0]
        self.__request_type_adapter = TypeAdapter(request_class)

    async def invoke(self, request: Request) -> Response:
        try:
            # Get request body
            body = await request.body()
            actual_request: TRequest = self.__request_type_adapter.validate_json(body)
            self.__logger.info(f"Full Request: <{actual_request}>")
            # Validate request
            await self._on_validate(actual_request)
            # Invoke handler
            actual_ok_response: TResponse = await self._on_invoke(actual_request);
            # Return OK response
            self.__logger.info(f"Full Response: <{HTTPStatus.OK} | {actual_ok_response}>")
            return self.__get_response(HTTPStatus.OK, actual_ok_response)
        except ManagedException as e:
            # Return error response
            actual_error_response: ErrorResponse = self._get_error_response(e)
            self.__logger.info(f"Full Response: <{e.status_code} | {actual_error_response}>")
            return self.__get_response(e.status_code, actual_error_response)
        except Exception as e:
            # Return error response
            actual_error_response: ErrorResponse = self._get_error_response(e)
            self.__logger.info(f"Full Response: <{HTTPStatus.INTERNAL_SERVER_ERROR} | {actual_error_response}>", exc_info=True)
            return self.__get_response(HTTPStatus.INTERNAL_SERVER_ERROR, actual_error_response)
    
    @abstractmethod
    async def _on_validate(self, request: TRequest) -> None:
        pass

    @abstractmethod
    async def _on_invoke(self, request: TRequest) -> TResponse:
        pass

    def _get_error_response(self, exception: Exception) -> ErrorResponse:
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