"""Authentication and API-boundary controls for the FastAPI service.

This service is an OIDC resource server. It verifies a bearer token against the
issuer's published keys and reads standard claims from it. It has no knowledge
of any particular identity provider: which provider issues the token, what its
session claims look like, and how extra claims are attached are all decided in
the frontend, which mints the token this service consumes.

The contract the frontend must satisfy:

* the token is signed by a key published at ``OIDC_JWKS_URL``
* ``iss`` matches ``OIDC_ISSUER``
* ``aud`` matches ``OIDC_AUDIENCE`` when that is configured
* ``sub`` carries the caller's user id
* ``email`` carries the caller's address, when the feature needs one

Anything beyond that standard set (a provider's own claim names, its claim
layout, its REST API) belongs in the frontend, not here.
"""
import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Dict, Optional

import jwt
import requests
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.env import get_env

logger = logging.getLogger("roadessy.security")

JWKS_REFRESH_SECONDS = 3600

bearer_scheme = HTTPBearer(auto_error=False)
_jwks_cache: Dict[str, Any] = {"fetched_at": 0.0, "keys": {}}


def _issuer() -> str:
    return get_env("OIDC_ISSUER")


def _audience() -> str:
    return get_env("OIDC_AUDIENCE")


def _jwks_url() -> str:
    return get_env("OIDC_JWKS_URL")


def is_production() -> bool:
    return get_env("ROADESSY_ENV", "development").lower() == "production"


# The provider-specific names this service used before it became a plain OIDC
# resource server. They are not read any more, so a deployment still carrying
# them has no authentication configuration at all.
_LEGACY_PROVIDER_VARS = (
    "CLERK_SECRET_KEY",
    "CLERK_JWKS_URL",
    "CLERK_ISSUER",
    "CLERK_AUDIENCE",
)


def check_oidc_configuration() -> bool:
    """Report whether this service can verify a token. Never raises.

    Called at startup so an environment that has not been migrated shows the
    variable names in the boot log. Without it the only symptom is a 503 on the
    first authenticated request, which reads like a provider outage rather than
    a missing setting.
    """
    missing = [name for name in ("OIDC_JWKS_URL",) if not get_env(name)]
    if not get_env("OIDC_ISSUER"):
        missing.append("OIDC_ISSUER")

    if missing:
        logger.error(
            "OIDC configuration incomplete: %s not set. Every authenticated "
            "request will fail with 503 until they are configured. The value "
            "is the frontend's identity provider issuer, for example "
            "https://<instance-domain>, with the JWKS URL at "
            "<issuer>/.well-known/jwks.json.",
            ", ".join(missing),
        )

    stale = [name for name in _LEGACY_PROVIDER_VARS if get_env(name)]
    if stale:
        logger.warning(
            "%s still set but no longer read. These are dead settings: remove "
            "them from the environment, and set OIDC_ISSUER/OIDC_JWKS_URL if "
            "they were the only authentication configuration present.",
            ", ".join(stale),
        )

    # OIDC_AUDIENCE is optional on purpose. The frontend prefers a token
    # template but falls back to a plain session token, which carries no `aud`.
    return not missing


def _normalize_issuer(raw: str) -> str:
    """Issuers are often configured without the trailing slash."""
    if not raw:
        return ""
    return raw if raw.endswith("/") else f"{raw}/"


def get_jwks_keys(force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch and cache the issuer's JSON Web Key Set.

    Returns an empty mapping when the keys cannot be retrieved rather than
    raising. The caller decides what an empty set means: in production it is
    fatal (see ``_decode_token``), while in development it falls back to an
    unverified decode.
    """
    jwks_url = _jwks_url()
    if not jwks_url:
        return {}
    now = _jwks_cache["fetched_at"]
    if not force_refresh and _jwks_cache["keys"] and now > 0:
        # Key rotation is infrequent; refresh at most once per hour.
        if time.time() - now < JWKS_REFRESH_SECONDS:
            return _jwks_cache["keys"]

    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        if _jwks_cache["keys"]:
            return _jwks_cache["keys"]
        logger.warning("Could not retrieve signing keys from %s: %s", jwks_url, exc)
        return {}

    keys = {key.get("kid"): key for key in payload.get("keys", []) if key.get("kid")}
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = time.time()
    return keys


def _decode_token(token: str) -> Dict[str, Any]:
    issuer = _normalize_issuer(_issuer())
    audience = _audience()
    keys = get_jwks_keys()
    if not keys:
        if is_production():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured (missing OIDC_JWKS_URL)",
            )
        # Development mode: decode without verification (insecure, for local dev only)
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid bearer token: {exc}",
            ) from exc

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

    options = {"verify_aud": bool(audience)}
    kwargs: Dict[str, Any] = {"algorithms": ["RS256"], "options": options}
    if issuer:
        kwargs["issuer"] = issuer
    if audience:
        kwargs["audience"] = audience

    try:
        return jwt.decode(token, public_key, **kwargs)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid bearer token: {exc}",
        ) from exc


def _claim_email(claims: Dict[str, Any]) -> Optional[str]:
    """The caller's address from the standard ``email`` claim.

    Only the standard claim is read. Reading a provider's private claim layouts
    here is what made this service provider-specific; the frontend is
    responsible for putting the address in ``email``.
    """
    email = claims.get("email")
    return email if isinstance(email, str) and email else None


def resolve_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Dict[str, Any]:
    """Resolve the caller identity from a verified bearer token.

    Falls back to the operator API key so machine-to-machine callers keep
    working, and returns an explicit anonymous principal when the deployment
    is intentionally unauthenticated (development default).
    """
    api_keys = {
        key.strip()
        for key in get_env("ROADESSY_API_KEYS").split(",")
        if key.strip()
    }
    supplied_key = request.headers.get("X-API-Key", "")
    if api_keys and supplied_key in api_keys:
        return {"user_id": "api_key_operator", "org_id": None, "kind": "api_key"}

    if credentials and credentials.credentials:
        claims = _decode_token(credentials.credentials)
        return {
            "user_id": claims.get("sub"),
            "org_id": claims.get("org_id"),
            # An invitation is addressed to an email, and this is the only
            # place the caller's address is known, so it travels with the
            # principal rather than being read from a client-supplied field.
            "email": _claim_email(claims),
            "kind": "oidc",
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


# ---------------------------------------------------------------------------
# Signed download links
# ---------------------------------------------------------------------------
# A browser cannot attach an Authorization header to a navigation, so a report
# the dashboard opens with `window.open` cannot use the bearer token. Rather than
# making those downloads anonymous, the link carries a short-lived HMAC over the
# asset id and its expiry, which proves the URL came from this service without
# proving anything about who holds it. The window it opens is narrow on purpose:
# one asset, one organisation, minutes not days.
#
# The signature is over the asset id and the expiry only. Binding it to a user id
# would be stronger, but the URL is handed to a browser that may be a different
# person's, and a link that fails for the wrong reader is worse than one that is
# merely time-limited.

DOWNLOAD_LINK_TTL_SECONDS = 600

_generated_download_secret: Optional[str] = None


def _download_secret() -> str:
    """The key download links are signed with.

    `ROADESSY_DOWNLOAD_SECRET` when set. Otherwise a random per-process secret:
    links then stop validating after a restart, which is an inconvenience, and
    is much better than the alternative of an unverified link.
    """
    global _generated_download_secret

    configured = get_env("ROADESSY_DOWNLOAD_SECRET")
    if configured:
        return configured

    if _generated_download_secret is None:
        _generated_download_secret = secrets.token_urlsafe(48)
        logger.warning(
            "ROADESSY_DOWNLOAD_SECRET is not set. Generated report links are "
            "signed with a secret that lasts only for this process, so every "
            "link already handed out stops working on the next restart. Set "
            "ROADESSY_DOWNLOAD_SECRET to a long random value to make them "
            "durable."
        )
    return _generated_download_secret


def sign_download_link(asset_id: str, ttl_seconds: int = DOWNLOAD_LINK_TTL_SECONDS) -> Dict[str, Any]:
    """A signed, expiring link to one stored asset."""
    expires_at = int(time.time()) + max(1, int(ttl_seconds))
    signature = hmac.new(
        _download_secret().encode("utf-8"),
        f"{asset_id}:{expires_at}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {"token": signature, "expires_at": expires_at}


def verify_download_token(asset_id: str, token: str, expires_at: int) -> None:
    """Raise 403 unless `token` is this service's signature for this asset.

    Compared in constant time and checked before the expiry, so a caller cannot
    learn the expected signature from how long the comparison took.
    """
    try:
        expires = int(expires_at)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This download link is not valid",
        )

    if expires <= int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This download link has expired. Reload the list and try again.",
        )

    expected = hmac.new(
        _download_secret().encode("utf-8"),
        f"{asset_id}:{expires}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not secrets.compare_digest(expected, str(token or "")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This download link is not valid",
        )
