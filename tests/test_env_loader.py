from __future__ import annotations

from pathlib import Path

from src.config.env import load_env_file


def test_load_env_file_accepts_utf8_bom_on_first_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("\ufeffTRADING_MODE=live\nSIGNAL_MODE=production\n", encoding="utf-8")

    values = load_env_file(env_path)

    assert values["TRADING_MODE"] == "live"
    assert values["SIGNAL_MODE"] == "production"
