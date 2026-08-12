from __future__ import annotations

import html
import json
import logging
from typing import Callable
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)

TelegramTransport = Callable[[str, dict[str, object]], object]


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: float = 10.0,
        transport: TelegramTransport | None = None,
    ) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_signal_table(self, table: str) -> bool:
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload: dict[str, object] = {
            "chat_id": self.chat_id,
            "text": (
                "<b>Binance Futures Rank2/Rank3 信号</b>\n"
                f"<pre>{html.escape(table)}</pre>"
            ),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            response = (
                self._transport(url, payload)
                if self._transport is not None
                else self._post_json(url, payload)
            )
            if not isinstance(response, dict) or response.get("ok") is not True:
                logger.warning("Telegram sendMessage returned an unsuccessful response.")
                return False
            return True
        except Exception as exc:
            logger.warning("Telegram signal notification failed: %s", exc)
            return False

    def _post_json(self, url: str, payload: dict[str, object]) -> object:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
