from __future__ import annotations

from src.config.settings import AppSettings
from src.notify.telegram import TelegramNotifier


SAMPLE_SIGNAL_TABLE = """SYMBOL  | RANK | PRICE |  GAIN |  V/R | STATUS
--------+------+-------+-------+------+-------
AGTUSDT |    2 |  0.12 | 94.5% | 6.20 | SKIP
SYNUSDT |    3 |  0.45 | 66.1% | 2.39 | SKIP

DETAILS
AGTUSDT: 超过上限，不交易
SYNUSDT: 60-80% 区间不交易"""


def main() -> None:
    settings = AppSettings.from_env_file()
    notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )
    if not notifier.enabled:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required in .env."
        )
    if not notifier.send_signal_table(SAMPLE_SIGNAL_TABLE):
        raise SystemExit("Telegram test message failed. Check token, chat ID, and network.")
    print("Telegram 测试消息发送成功。")


if __name__ == "__main__":
    main()
