"""The boot check has to name the right variables.

Renaming the provider settings to provider-neutral OIDC ones is only done in the
hosted environments, and a deployment that has not been renamed has no
authentication configuration at all. The only other symptom is a 503 on the first
authenticated request, so the boot log is where the settings get found. These
tests pin the two behaviours that matter: a missing setting is reported by name,
and leftover provider-specific settings are called out as dead.
"""

import logging

import pytest

from app.core import security


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "OIDC_ISSUER",
        "OIDC_JWKS_URL",
        "OIDC_AUDIENCE",
        "ROADESSY_ENV",
        *security._LEGACY_PROVIDER_VARS,
    ):
        monkeypatch.delenv(name, raising=False)


def _configure(issuer="https://issuer.example", jwks="https://issuer.example/jwks"):
    import os

    if issuer:
        os.environ["OIDC_ISSUER"] = issuer
    if jwks:
        os.environ["OIDC_JWKS_URL"] = jwks


class TestReport:
    def test_a_fully_configured_service_reports_ready(self):
        _configure()

        assert security.check_oidc_configuration() is True

    def test_a_missing_jwks_url_is_reported_by_name(self, caplog):
        _configure(jwks=None)

        with caplog.at_level(logging.ERROR):
            assert security.check_oidc_configuration() is False

        assert "OIDC_JWKS_URL" in caplog.text

    def test_a_missing_issuer_is_reported_by_name(self, caplog):
        _configure(issuer=None)

        with caplog.at_level(logging.ERROR):
            assert security.check_oidc_configuration() is False

        assert "OIDC_ISSUER" in caplog.text

    def test_a_service_with_no_oidc_settings_at_all_is_reported(self, caplog):
        """The state a deployment still carrying only the old names is in."""
        with caplog.at_level(logging.ERROR):
            assert security.check_oidc_configuration() is False

        assert "OIDC_JWKS_URL" in caplog.text
        assert "OIDC_ISSUER" in caplog.text


class TestAudience:
    def test_audience_is_optional(self):
        """The frontend falls back to a session token, which has no `aud`."""
        _configure()

        assert security.check_oidc_configuration() is True


class TestLegacySettings:
    def test_leftover_provider_settings_are_called_out_as_dead(self, caplog):
        import os

        _configure()
        os.environ["CLERK_SECRET_KEY"] = "sk_live_example"

        with caplog.at_level(logging.WARNING):
            security.check_oidc_configuration()

        assert "CLERK_SECRET_KEY" in caplog.text
        assert "no longer read" in caplog.text

    def test_leftover_settings_do_not_make_a_good_service_look_broken(self):
        import os

        _configure()
        os.environ["CLERK_JWKS_URL"] = "https://clerk.example/jwks"

        assert security.check_oidc_configuration() is True
