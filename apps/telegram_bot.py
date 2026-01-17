from __future__ import annotations

import os
from typing import Iterable

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from utils.prediction_service import available_datasets, build_prediction


def format_datasets(datasets: Iterable[str]) -> str:
    return "\n".join(f"• {dataset}" for dataset in datasets)


def list_available_pairs() -> list[str]:
    return [f"{item.symbol} {item.interval}" for item in available_datasets()]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = (
        "Welcome to CryptoPredictions!\n\n"
        "Use /predict SYMBOL INTERVAL MODEL to get a forecast.\n"
        "Example: /predict ETHUSD 1d random_forest\n\n"
        "Use /datasets to see available local datasets."
    )
    await update.message.reply_text(message)


async def datasets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pairs = list_available_pairs()
    if not pairs:
        await update.message.reply_text("No datasets found in the ./data folder.")
        return

    await update.message.reply_text(
        "Available datasets:\n" + format_datasets(pairs)
    )


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /predict SYMBOL INTERVAL [MODEL]\n"
            "Example: /predict ETHUSD 1d random_forest"
        )
        return

    symbol = context.args[0].upper()
    interval = context.args[1].lower()
    model_type = context.args[2].lower() if len(context.args) > 2 else "random_forest"

    try:
        result = build_prediction(symbol, interval, model_type=model_type)
    except FileNotFoundError:
        await update.message.reply_text(
            "Dataset not found. Use /datasets to see available symbols."
        )
        return
    except Exception as exc:  # pragma: no cover - guard rail for bot replies
        await update.message.reply_text(f"Prediction failed: {exc}")
        return

    message = (
        f"*{result.symbol}* ({result.interval})\n"
        f"Model: `{result.model}`\n"
        f"Predicted mean: *${result.predicted_mean:,.2f}*\n"
        f"Last close: ${result.last_close:,.2f} on {result.last_timestamp}"
    )
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("datasets", datasets))
    app.add_handler(CommandHandler("predict", predict))

    app.run_polling()


if __name__ == "__main__":
    main()
