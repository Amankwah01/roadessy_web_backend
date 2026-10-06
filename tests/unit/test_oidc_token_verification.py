"""A token the way an identity provider issues one must verify.

The issuer comparison is an exact string match, so it is sensitive to a
difference nobody can see in the configuration: a trailing slash. That is not
hypothetical. `_normalize_issuer` used to *append* a slash to the configured
issuer, so it expected `https://host/` while Clerk issues `iss:
https://host`, and PyJWT rejected every real token with "Invalid issuer". Every
authenticated request answered 401, and the frontend reported it as nothing more
than `HTTP 401`, which is why it read as an unexplained failure rather than as a
verifier bug.

These tests mint their own tokens and inject the matching key, so they exercise
the real verification path without an identity provider.
"""
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app.core import security

ISSUER = "https://issuer.example"


@pytest.fixture
def signing_key(monkeypatch):
    """A private key the tests sign with, published to the module's key cache."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    jwk = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk["kid"] = "ins_test"
    jwk["alg"] = "RS256"

    monkeypatch.setitem(security._jwks_cache, "keys", {"ins_test": jwk})
    monkeypatch.setitem(security._jwks_cache, "fetched_at", time.time())
    return private_pem


def _token(private_pem, **claims):
    payload = {"sub": "user_2abc", "sid": "sess_1", "exp": int(time.time()) + 300, **claims}
    return jwt.encode(
        payload, private_pem, algorithm="RS256", headers={"kid": "ins_test"}
    )


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_JWKS_URL", f"{ISSUER}/.well-known/jwks.json")
    monkeypatch.delenv("OIDC_AUDIENCE", raising=False)


class TestTheIssuerSpelling:
    def test_the_normalizer_does_not_invent_a_trailing_slash(self):
        assert security._normalize_issuer(ISSUER) == ISSUER

    def test_a_configured_trailing_slash_is_reconciled_away(self):
        """Both spellings of the setting have to work against one token form."""
        assert security._normalize_issuer(f"{ISSUER}/") == ISSUER

    @pytest.mark.parametrize("configured", [ISSUER, f"{ISSUER}/"])
    def test_a_token_from_the_provider_verifies(self, signing_key, configured, monkeypatch):
        """The whole point: a real-shaped token is accepted.

        Clerk issues `iss` with no trailing slash, so that is what the token
        carries here regardless of how the setting is spelled.
        """
        monkeypatch.setenv("OIDC_ISSUER", configured)

        decoded = security._decode_token(_token(signing_key, iss=ISSUER))

        assert decoded["sub"] == "user_2abc"


class TestTheVerifierStillRefuses:
    def test_a_token_from_another_issuer_is_refused(self, signing_key):
        with pytest.raises(Exception) as caught:
            security._decode_token(_token(signing_key, iss="https://evil.example"))

        assert caught.value.status_code == 401
        assert "issuer" in caught.value.detail.lower()

    def test_a_token_signed_by_an_unknown_key_is_refused(self, signing_key):
        token = jwt.encode(
            {"sub": "user_2abc", "iss": ISSUER, "exp": int(time.time()) + 300},
            signing_key,
            algorithm="RS256",
            headers={"kid": "ins_rotated_out"},
        )

        with pytest.raises(Exception) as caught:
            security._decode_token(token)

        assert caught.value.status_code == 401

    def test_an_expired_token_is_refused(self, signing_key):
        with pytest.raises(Exception) as caught:
            security._decode_token(
                _token(signing_key, iss=ISSUER, exp=int(time.time()) - 60)
            )

        assert caught.value.status_code == 401

    def test_a_tampered_token_is_refused(self, signing_key):
        """Signature verification is the point of the whole exercise."""
        token = _token(signing_key, iss=ISSUER)
        header, payload, signature = token.split(".")
        tampered = f"{header}.{payload}.{signature[:-4]}AAAA"

        with pytest.raises(Exception) as caught:
            security._decode_token(tampered)

        assert caught.value.status_code == 401