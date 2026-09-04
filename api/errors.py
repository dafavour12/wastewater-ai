from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def build_error_response(
    error: str,
    message: str,
    status_code: int,
    details: Any = None,
    detail: Any = None,
) -> JSONResponse:
    content = {
        "error": error,
        "message": message,
        "status_code": status_code,
    }

    if detail is not None:
        content["detail"] = detail

    if details is not None:
        content["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    detail = exc.detail

    if isinstance(detail, str):
        message = detail
    else:
        message = "The request could not be completed."

    return build_error_response(
        error="http_error",
        message=message,
        status_code=exc.status_code,
        detail=detail,
        details=detail if not isinstance(detail, str) else None,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()

    return build_error_response(
        error="validation_error",
        message="Request validation failed.",
        status_code=422,
        detail=errors,
        details=errors,
    )


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return build_error_response(
        error="internal_server_error",
        message="An unexpected internal server error occurred.",
        status_code=500,
    )
