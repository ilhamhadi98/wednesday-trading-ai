from __future__ import annotations

import json

from trading_bot.config import OUTPUT_DIR, ensure_output_dir, load_settings
from trading_bot.modeling import save_artifacts, save_training_history
from trading_bot.mt5_client import MT5Client
from trading_bot.workflows import fetch_feature_frame, train_once


def main() -> None:
    settings = load_settings()
    ensure_output_dir()
    client = MT5Client(settings)

    try:
        client.connect()
        frame, feature_cols = fetch_feature_frame(client, settings)
        model, scaler, history, metrics, dataset = train_once(
            frame=frame,
            feature_cols=feature_cols,
            settings=settings,
        )

        save_artifacts(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            lookback=settings.lookback,
        )
        history_path = save_training_history(history)
        metrics_path = OUTPUT_DIR / "train_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("Training selesai.")
        print(f"Jumlah sample sequence: {len(dataset.x)}")
        print(f"Metrics validasi: {metrics}")
        print(f"Model tersimpan: {OUTPUT_DIR / 'model.keras'}")
        print(f"Scaler tersimpan: {OUTPUT_DIR / 'scaler.pkl'}")
        print(f"History tersimpan: {history_path}")
        print(f"Metrics tersimpan: {metrics_path}")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
