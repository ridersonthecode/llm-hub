"""Startpunkt für 'python -m vllm_manager' (das der systemd-Dienst vllm.service nutzt)."""
import uvicorn

from .config import get_config


def main() -> None:
    cfg = get_config()
    uvicorn.run("vllm_manager.main:app", host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
