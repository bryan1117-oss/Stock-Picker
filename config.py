import os
from dataclasses import dataclass


def _secret(key: str, default: str = "") -> str:
    """Read a config value from Streamlit's secrets manager when running on Streamlit
    Community Cloud (st.secrets), falling back to an environment variable for local/dev
    runs, then a default. Wrapped in try/except because st.secrets raises if no
    secrets.toml exists at all (the normal case for local development)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


@dataclass
class Config:
    # The SEC REQUIRES a descriptive User-Agent with a real contact email on every
    # request to data.sec.gov. Requests without one get rate-limited or blocked outright.
    # Local: export SEC_USER_AGENT="your-name your-email@example.com"
    # Streamlit Cloud: set SEC_USER_AGENT in the app's Settings -> Secrets.
    sec_user_agent: str = _secret("SEC_USER_AGENT", "")

    # Where cached SEC JSON responses are stored. Cached facts live 6h, the ticker map 24h.
    # On Streamlit Cloud this is ephemeral and resets on redeploy -- that's fine, it's
    # only a performance cache, not a data store.
    cache_dir: str = _secret("CACHE_DIR", ".cache")

    # Optional: only needed if you enable the AI interpretation checkbox.
    openai_api_key: str = _secret("OPENAI_API_KEY", "")


CONFIG = Config()

if not CONFIG.sec_user_agent:
    import warnings
    warnings.warn(
        "SEC_USER_AGENT is not set. The SEC requires a contact email in the User-Agent "
        "header and may throttle or block requests without one. "
        'Set it locally with: export SEC_USER_AGENT="your-name your-email@example.com" '
        "or, on Streamlit Cloud, in Settings -> Secrets."
    )

