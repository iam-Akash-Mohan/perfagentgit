"""MySQLTarget: the profile is the source of truth, `from_env` is an override.

No field here carries a default (see the class docstring) -- these tests
cover the two ways a target gets built for real, and the two ways building
one refuses instead of guessing.
"""

from __future__ import annotations

import pytest

from senapathy.contracts.repo import DatabaseKind, DatabaseProfile
from senapathy.errors import ToolError
from senapathy.measure.sql import MySQLTarget


def _admin_env(monkeypatch: pytest.MonkeyPatch, user: str = "admin",
               password: str = "adminpw") -> None:
    monkeypatch.setenv("SENAPATHY_MYSQL_ADMIN_USER", user)
    monkeypatch.setenv("SENAPATHY_MYSQL_ADMIN_PASSWORD", password)


# ──────────────────────────────  from_profile  ────────────────────────────


def test_from_profile_resolves_host_port_schema_from_split_fields(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Onboarding already splits the jdbc_url it found -- when it did, those
    fields win outright and the admin login comes from the environment."""
    _admin_env(monkeypatch)
    database = DatabaseProfile(
        kind=DatabaseKind.MYSQL,
        jdbc_url="jdbc:mysql://db.internal:3306/storefront",
        host="db.internal", port=3306, database="storefront",
        user="app_user", password="app_secret",
    )

    target = MySQLTarget.from_profile(database)

    assert (target.host, target.port, target.database) == (
        "db.internal", 3306, "storefront")
    # The admin login, never the app's own detected credentials.
    assert (target.user, target.password) == ("admin", "adminpw")


def test_from_profile_falls_back_to_parsing_the_jdbc_url(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DatabaseProfile that only carries the raw URL -- host/port/database
    not split out -- still resolves, by parsing it here."""
    _admin_env(monkeypatch)
    database = DatabaseProfile(
        kind=DatabaseKind.MYSQL,
        jdbc_url="jdbc:mysql://db.internal:3306/storefront?useSSL=false",
    )

    target = MySQLTarget.from_profile(database)

    assert (target.host, target.port, target.database) == (
        "db.internal", 3306, "storefront")


def test_from_profile_parses_a_jdbc_url_with_no_explicit_port(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _admin_env(monkeypatch)
    database = DatabaseProfile(
        kind=DatabaseKind.MYSQL, jdbc_url="jdbc:mysql://db.internal/storefront",
    )

    target = MySQLTarget.from_profile(database)

    assert target.host == "db.internal"
    assert target.database == "storefront"


def test_from_profile_refuses_a_non_mysql_kind_by_name() -> None:
    database = DatabaseProfile(kind=DatabaseKind.POSTGRES)

    with pytest.raises(ToolError, match="not mysql"):
        MySQLTarget.from_profile(database)


def test_from_profile_refuses_when_host_port_or_schema_cannot_be_found(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _admin_env(monkeypatch)
    database = DatabaseProfile(kind=DatabaseKind.MYSQL)  # nothing detected

    with pytest.raises(ToolError, match="missing host, port, database"):
        MySQLTarget.from_profile(database)


def test_from_profile_refuses_by_name_when_the_admin_login_is_absent(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admin login is never the app's own credentials -- its absence
    must be refused by name, not quietly filled with root/root.

    `from_profile` calls `load_dotenv()`, same as `make_client` does for API
    keys -- a real `.env` on the developer's machine would otherwise refill
    the variable the moment `delenv` cleared it, exactly the way a stray
    `CEREBRAS_API_KEY` in `.env` once made a provider-switch test pass for
    the wrong reason (see test_llm.py)."""
    import senapathy.measure.sql as sql_mod

    monkeypatch.setattr(sql_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("SENAPATHY_MYSQL_ADMIN_USER", raising=False)
    monkeypatch.delenv("SENAPATHY_MYSQL_ADMIN_PASSWORD", raising=False)
    database = DatabaseProfile(
        kind=DatabaseKind.MYSQL, host="db.internal", port=3306,
        database="storefront", user="app_user", password="app_secret",
    )

    with pytest.raises(ToolError, match="SENAPATHY_MYSQL_ADMIN_USER") as excinfo:
        MySQLTarget.from_profile(database)
    assert "SENAPATHY_MYSQL_ADMIN_PASSWORD" in str(excinfo.value)
    # Never silently reuses the app's own detected login.
    assert "app_user" not in str(excinfo.value)


# ────────────────────────────────  from_env  ──────────────────────────────


def test_from_env_requires_every_variable_and_refuses_by_name(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE"):
        monkeypatch.delenv(f"SENAPATHY_MYSQL_{name}", raising=False)

    with pytest.raises(ToolError) as excinfo:
        MySQLTarget.from_env()
    message = str(excinfo.value)
    for name in ("SENAPATHY_MYSQL_HOST", "SENAPATHY_MYSQL_PORT",
                "SENAPATHY_MYSQL_USER", "SENAPATHY_MYSQL_PASSWORD",
                "SENAPATHY_MYSQL_DATABASE"):
        assert name in message


def test_from_env_builds_a_target_when_every_variable_is_set(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SENAPATHY_MYSQL_HOST", "127.0.0.1")
    monkeypatch.setenv("SENAPATHY_MYSQL_PORT", "3307")
    monkeypatch.setenv("SENAPATHY_MYSQL_USER", "root")
    monkeypatch.setenv("SENAPATHY_MYSQL_PASSWORD", "root")
    monkeypatch.setenv("SENAPATHY_MYSQL_DATABASE", "mini")

    target = MySQLTarget.from_env()

    assert (target.host, target.port, target.user, target.password,
           target.database) == ("127.0.0.1", 3307, "root", "root", "mini")


# ───────────────  no field on the model defaults by itself  ──────────────


def test_no_field_has_a_default() -> None:
    """The regression this whole module exists to prevent: a default that
    happens to be right for one repo."""
    for name, field in MySQLTarget.model_fields.items():
        assert field.is_required(), (
            f"MySQLTarget.{name} has a default ({field.default!r}); a "
            "connection target must come from from_profile() or from_env(), "
            "never a value baked into the model."
        )
