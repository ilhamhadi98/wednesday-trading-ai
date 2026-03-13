from __future__ import annotations

import time

import numpy as np

from trading_bot.config import load_settings
from trading_bot.data_pipeline import build_multitimeframe_frame
from trading_bot.execution import DemoExecutor
from trading_bot.modeling import load_artifacts, predict_proba, save_artifacts
from trading_bot.mt5_client import MT5Client
from trading_bot.workflows import fetch_feature_frame, train_once


def signal_from_probability(prob: float, buy_th: float, sell_th: float) -> int:
    if prob >= buy_th:
        return 1
    if prob <= sell_th:
        return -1
    return 0


def retrain_live_model(client: MT5Client):
    settings = load_settings()
    frame, feature_cols = fetch_feature_frame(
        client,
        settings,
        fast_tf=settings.live_fast_timeframe,
        slow_tf=settings.live_slow_timeframe,
    )
    model, scaler, history, metrics, _ = train_once(
        frame=frame,
        feature_cols=feature_cols,
        settings=settings,
    )
    save_artifacts(model, scaler, feature_cols, settings.lookback)
    return model, scaler, feature_cols, history, metrics


def main() -> None:
    settings = load_settings()
    client = MT5Client(settings)
    last_seen_bar_time = None
    bars_since_retrain = 0

    try:
        client.connect()
        spec = client.symbol_spec(settings.symbol)
        artifacts = load_artifacts()
        executor = DemoExecutor(client, settings, spec)

        print(
            f"[LIVE] start symbol={settings.symbol}, buy_th={settings.prediction_buy_threshold}, "
            f"sell_th={settings.prediction_sell_threshold}"
            ,
            flush=True,
        )
        print("[LIVE] mode demo aktif. Tekan CTRL+C untuk berhenti.", flush=True)

        while True:
            h1 = client.copy_rates(
                settings.symbol, settings.live_fast_timeframe, settings.bars_h1
            )
            h4 = client.copy_rates(
                settings.symbol, settings.live_slow_timeframe, settings.bars_h4
            )
            frame, _ = build_multitimeframe_frame(h1, h4)
            if len(frame) < 5:
                time.sleep(settings.live_sleep_seconds)
                continue

            # Gunakan candle yang sudah close, hindari candle berjalan.
            closed_frame = frame.iloc[:-1].copy()
            if len(closed_frame) < artifacts.lookback:
                print("[LIVE] data belum cukup untuk prediksi.", flush=True)
                time.sleep(settings.live_sleep_seconds)
                continue

            current_bar = closed_frame.index[-1]
            if last_seen_bar_time is not None and current_bar <= last_seen_bar_time:
                time.sleep(settings.live_sleep_seconds)
                continue

            needed_cols = artifacts.feature_cols
            missing_cols = [c for c in needed_cols if c not in closed_frame.columns]
            if missing_cols:
                raise RuntimeError(f"Feature hilang saat live: {missing_cols}")

            window = closed_frame.iloc[-artifacts.lookback :].copy()
            x = artifacts.scaler.transform(window[needed_cols].values)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            prob = float(predict_proba(artifacts.model, x)[0])
            signal = signal_from_probability(
                prob, settings.prediction_buy_threshold, settings.prediction_sell_threshold
            )
            action = executor.handle_signal(signal)
            side = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
            print(
                f"[LIVE] bar={current_bar} prob={prob:.4f} signal={side} action_ok={action.ok} msg={action.message}",
                flush=True,
            )

            bars_since_retrain += 1
            if bars_since_retrain >= settings.retrain_every_bars:
                print("[LIVE] retraining model dari data terbaru...", flush=True)
                model, scaler, feature_cols, _, metrics = retrain_live_model(client)
                artifacts.model = model
                artifacts.scaler = scaler
                artifacts.feature_cols = feature_cols
                bars_since_retrain = 0
                print(f"[LIVE] retraining selesai. metrics={metrics}", flush=True)

            last_seen_bar_time = current_bar
            time.sleep(settings.poll_seconds)

    except KeyboardInterrupt:
        print("[LIVE] dihentikan oleh user.", flush=True)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
