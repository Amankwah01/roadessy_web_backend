"""Small, deployable API-boundary controls for the FastAPI service."""
import os

from fastapi import HTTPException, Request, status


def is_production() -> bool:
    return os.getenv("ROADESSY_ENV", "development").lower() == "production"


async def require_api_key(request: Request) -> None:
    """Require an operator-provided API key in production deployments.

    Keys are configured as a comma-separated ROADESSY_API_KEYS environment
    variable. Authentication is deliberately disabled in development so local
    OpenAPI and test workflows remain usable.
    """
    if not is_production():
        return
    configured = {key.strip() for key in os.getenv("ROADESSY_API_KEYS", "").split(",") if key.strip()}
    supplied = request.headers.get("X-API-Key", "")
    if not configured or supplied not in configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="A valid X-API-Key is required")
