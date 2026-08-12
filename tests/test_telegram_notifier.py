from __future__ import annotations

from src.notify.telegram import TelegramNotifier


def test_telegram_notifier_sends_signal_table_as_preformatted_html() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def transport(url: str, payload: dict[str, object]) -> object:
        calls.append((url, payload))
        return {"ok": True, "result": {"message_id": 1}}

    notifier = TelegramNotifier(
        bot_token="123:token",
        chat_id="456",
        transport=transport,
    )

    sent = notifier.send_signal_table(
        "信号 | 排名 | 价格\nAGTUSDT | 2 | 0.12"
    )

    assert sent is True
    assert calls == [
        (
            "https://api.telegram.org/bot123:token/sendMessage",
            {
                "chat_id": "456",
                "text": (
                    "<b>Binance Futures Rank2/Rank3 信号</b>\n"
                    "<pre>信号 | 排名 | 价格\nAGTUSDT | 2 | 0.12</pre>"
                ),
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )
    ]


def test_telegram_notifier_is_disabled_when_credentials_are_blank() -> None:
    notifier = TelegramNotifier(bot_token="", chat_id="")

    assert notifier.enabled is False
    assert notifier.send_signal_table("table") is False


def test_telegram_failure_returns_false_without_raising() -> None:
    def transport(url: str, payload: dict[str, object]) -> object:
        raise OSError("network unavailable")

    notifier = TelegramNotifier(
        bot_token="123:token",
        chat_id="456",
        transport=transport,
    )

    assert notifier.send_signal_table("table") is False
