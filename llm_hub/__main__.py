"""Startpunkt für 'python -m llm_hub' (das der systemd-Dienst llm-hub.service nutzt)."""
import uvicorn

from .config import get_config


def main() -> None:
    cfg = get_config()
    uvicorn.run("llm_hub.main:app", host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
