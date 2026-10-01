"""Authentication and API-boundary controls for the FastAPI service."""
import os
import time
from typing import Any, Dict, Optional

import jwt
import requests
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

CLERK_ISSUER = os.getenv("CLERK_ISSUER", "").strip()
CLERK_AUDIENCE = os.getenv("CLERK_AUDIENCE", "").strip()
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL", "").strip()

JWKS_REFRESH_SECONDS = 3600

bearer_scheme = HTTPBearer(auto_error=False)
_jwks_cache: Dict[str, Any] = {"fetched_at": 0.0, "keys": {}}


def is_production() -> bool:
    return os.getenv("ROADESSY_ENV", "development").lower() == "production"


def _normalize_issuer(raw: str) -> str:
    """Clerk issuers are sometimes stored without the trailing slash."""
    if not raw:
        return ""
    return raw if raw.endswith("/") else f"{raw}/"


def get_jwks_keys(force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch and cache Clerk's JSON Web Key Set."""
    if not CLERK_JWKS_URL:
        return {}
    now = _jwks_cache["fetched_at"]
    if not force_refresh and _jwks_cache["keys"] and now > 0:
        # JWKS rotation is infrequent; refresh at most once per hour.
        if time.time() - now < JWKS_REFRESH_SECONDS:
            return _jwks_cache["keys"]

    try:
        response = requests.get(CLERK_JWKS_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        if _jwks_cache["keys"]:
            return _jwks_cache["keys"]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to retrieve signing keys: {exc}",
        ) from exc

    keys = {key.get("kid"): key for key in payload.get("keys", []) if key.get("kid")}
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = time.time()
    return keys


def _decode_clerk_token(token: str) -> Dict[str, Any]:
    issuer = _normalize_issuer(CLERK_ISSUER)
    keys = get_jwks_keys()
    if not keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured (missing CLERK_JWKS_URL)",
        )

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed bearer token",
        ) from exc

    jwk = keys.get(header.get("kid"))
    if jwk is None:
        keys = get_jwks_keys(force_refresh=True)
        jwk = keys.get(header.get("kid"))
    if jwk is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown signing key",
        )

    from jwt.algorithms import RSAAlgorithm

    public_key = RSAAlgorithm.from_jwk(jwk)

    options = {"verify_aud": bool(CLERK_AUDIENCE)}
    kwargs: Dict[str, Any] = {"algorithms": ["RS256"], "options": options}
    if issuer:
        kwargs["issuer"] = issuer
    if CLERK_AUDIENCE:
        kwargs["audience"] = CLERK_AUDIENCE

    try:
        return jwt.decode(token, public_key, **kwargs)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid bearer token: {exc}",
        ) from exc


def _claim_email(claims: Dict[str, Any]) -> Optional[str]:
    """The caller's email from the token, wherever the provider put it.

    Claim layout has changed across provider versions and across token types
    (session vs OAuth), so this checks the places it has appeared rather than
    one. Returns None rather than guessing when none is present.
    """
    direct = claims.get("email")
    if isinstance(direct, str) and direct:
        return direct

    primary = claims.get("primary_email_address")
    if isinstance(primary, str) and primary:
        return primary

    for key in ("email_address", "email_addresses", "emails"):
        value = claims.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first:
                return first
            if isinstance(first, dict):
                address = first.get("email_address") or first.get("email")
                if isinstance(address, str) and address:
                    return address
    return None


def resolve_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Dict[str, Any]:
    """Resolve the caller identity from a Clerk session token.

    Falls back to the operator API key so machine-to-machine callers keep
    working, and returns an explicit anonymous principal when the deployment
    is intentionally unauthenticated (development default).
    """
    api_keys = {
        key.strip()
        for key in os.getenv("ROADESSY_API_KEYS", "").split(",")
        if key.strip()
    }
    supplied_key = request.headers.get("X-API-Key", "")
    if api_keys and supplied_key in api_keys:
        return {"user_id": "api_key_operator", "org_id": None, "kind": "api_key"}

    if credentials and credentials.credentials:
        claims = _decode_clerk_token(credentials.credentials)
        return {
            "user_id": claims.get("sub"),
            "org_id": claims.get("org_id"),
            # An invitation is addressed to an email, and this is the only
            # place the caller's address is known, so it travels with the
            # principal rather than being read from a client-supplied field.
            "email": _claim_email(claims),
            "kind": "clerk",
        }

    if is_production():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required",
        )

    return {"user_id": None, "org_id": None, "kind": "anonymous"}


async def get_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    return resolve_principal(request, credentials)


def require_authenticated_user(
    principal: Dict[str, Any] = Depends(get_principal),
) -> str:
    """Return the caller's verified user id, rejecting anonymous callers."""
    user_id = principal.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user_id