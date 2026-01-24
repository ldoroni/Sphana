from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Type, get_args
from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from managed_exceptions import ManagedException, InternalErrorException
from pydantic import TypeAdapter
from request_handler.error_response import ErrorResponse

class RequestHandler[TRequest, TResponse](ABC):
    def __init__(self):
        base_type = self.__class__.__orig_bases__[0] # type: ignore
        generics = get_args(base_type)
        request_class = generics[0]
        self.request_type_adapter = TypeAdapter(request_class)

    async def invoke(self, request: Request) -> Response:
        try:
            # Get request body
            body = await request.body()
            actual_request: TRequest = self.request_type_adapter.validate_json(body)
            # Validate request
            await self._on_validate(actual_request)
            # Invoke handler
            actualResponse: TResponse = await self._on_invoke(actual_request);
            # Return OK response
            return JSONResponse(
                status_code=HTTPStatus.OK,
                content=jsonable_encoder(actualResponse)
            )
        except ManagedException as e:
            # Return error response
            return JSONResponse(
                status_code=e.status_code,
                content=jsonable_encoder(self._get_error_response(e))
            )
        except Exception as e:
            # Return error response
            return JSONResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                content=jsonable_encoder(self._get_error_response(e))
            )
    
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