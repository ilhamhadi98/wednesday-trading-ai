from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model.keras"
SCALER_PATH = OUTPUT_DIR / "scaler.pkl"
BACKTEST_REPORT_PATH = OUTPUT_DIR / "backtest_report.json"
BACKTEST_TRADES_PATH = OUTPUT_DIR / "backtest_trades.csv"
MULTI_BACKTEST_TRADES_PATH = OUTPUT_DIR / "multi_backtest_trades.csv"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


@dataclass
class Settings:
    mt5_login: int | None
    mt5_password: str | None
    mt5_server: str | None
    mt5_path: str | None

    symbol: str
    symbols: list[str]
    symbol_discovery_mode: str
    pair_filter_mode: str
    pair_include: list[str]
    pair_exclude: list[str]
    pair_max_volume_min: float
    scan_max_symbols: int
    min_bars_required: int
    max_signal_bar_age_hours: int
    max_active_pairs: int
    risk_profile: str
    bars_h1: int
    bars_h4: int
    bars_m15: int
    bars_m1: int
    backtest_fast_timeframe: str
    backtest_slow_timeframe: str
    live_fast_timeframe: str
    live_slow_timeframe: str
    lookback: int
    retrain_every_bars: int

    epochs: int
    batch_size: int
    learning_rate: float
    prediction_buy_threshold: float
    prediction_sell_threshold: float

    initial_balance: float
    risk_per_trade: float
    fixed_lot: float
    use_risk_position_sizing: bool
    stop_loss_pips: float
    take_profit_pips: float
    # Trailing stop: aktifkan dengan True, trailing_stop_pips = jarak trailing dari harga
    use_trailing_stop: bool
    trailing_stop_pips: float
    # Circuit breaker: hentikan entry baru jika drawdown sudah melebihi batas ini (0–1)
    max_drawdown_circuit_breaker: float
    # Batas maksimal consecutive losses sebelum berhenti entry
    max_consecutive_losses: int
    max_spread_points: int
    deviation: int
    magic_number: int

    poll_seconds: int
    live_sleep_seconds: int

    # Ollama LLM Multi-Agent settings
    ollama_base_url: str
    ollama_news_model: str
    ollama_tech_model: str
    ollama_decision_model: str
    ollama_enabled: bool
    ollama_timeout: int


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    mt5_login = os.getenv("MT5_LOGIN")
    settings = Settings(
        mt5_login=int(mt5_login) if mt5_login else None,
        mt5_password=os.getenv("MT5_PASSWORD"),
        mt5_server=os.getenv("MT5_SERVER"),
        mt5_path=os.getenv("MT5_PATH"),
        symbol=os.getenv("SYMBOL", "EURUSD"),
        symbols=_env_list("SYMBOLS", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]),
        symbol_discovery_mode=os.getenv("SYMBOL_DISCOVERY_MODE", "VISIBLE"),
        pair_filter_mode=os.getenv("PAIR_FILTER_MODE", "ALL"),
        pair_include=_env_list("PAIR_INCLUDE", []),
        pair_exclude=_env_list("PAIR_EXCLUDE", []),
        pair_max_volume_min=_env_float("PAIR_MAX_VOLUME_MIN", 1.0),
        scan_max_symbols=_env_int("SCAN_MAX_SYMBOLS", 50),
        min_bars_required=_env_int("MIN_BARS_REQUIRED", 800),
        max_signal_bar_age_hours=_env_int("MAX_SIGNAL_BAR_AGE_HOURS", 168),
        max_active_pairs=_env_int("MAX_ACTIVE_PAIRS", 3),
        risk_profile=os.getenv("RISK_PROFILE", "SEDANG"),
        bars_h1=_env_int("BARS_H1", 10000),
        bars_h4=_env_int("BARS_H4", 5000),
        # M15 membutuhkan lebih banyak bar (46 bulan * ~30 hari * ~96 bar/hari ≈ 132.480)
        bars_m15=_env_int("BARS_M15", 135000),
        bars_m1=_env_int("BARS_M1", 100000),
        backtest_fast_timeframe=os.getenv("BACKTEST_FAST_TF", "H1"),
        backtest_slow_timeframe=os.getenv("BACKTEST_SLOW_TF", "H4"),
        live_fast_timeframe=os.getenv("LIVE_FAST_TF", "H1"),
        live_slow_timeframe=os.getenv("LIVE_SLOW_TF", "H4"),
        lookback=_env_int("LOOKBACK", 96),
        retrain_every_bars=_env_int("RETRAIN_EVERY_BARS", 24),
        epochs=_env_int("EPOCHS", 20),
        batch_size=_env_int("BATCH_SIZE", 64),
        learning_rate=_env_float("LEARNING_RATE", 0.001),
        # Threshold lebih ketat untuk mengurangi sinyal palsu & loss
        prediction_buy_threshold=_env_float("PRED_BUY_THRESHOLD", 0.60),
        prediction_sell_threshold=_env_float("PRED_SELL_THRESHOLD", 0.40),
        # Modal awal diubah ke 100 sesuai permintaan
        initial_balance=_env_float("INITIAL_BALANCE", 100.0),
        risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
        fixed_lot=_env_float("FIXED_LOT", 0.01),
        use_risk_position_sizing=_env_bool("USE_RISK_POSITION_SIZING", True),
        # SL lebih ketat untuk M15 (20 pip); TP 40 pip → rasio R:R = 1:2
        stop_loss_pips=_env_float("STOP_LOSS_PIPS", 20.0),
        take_profit_pips=_env_float("TAKE_PROFIT_PIPS", 40.0),
        # Trailing stop: lindungi keuntungan yang sudah berjalan
        use_trailing_stop=_env_bool("USE_TRAILING_STOP", True),
        trailing_stop_pips=_env_float("TRAILING_STOP_PIPS", 12.0),
        # Circuit breaker: hentikan entry baru jika drawdown >= 15%
        max_drawdown_circuit_breaker=_env_float("MAX_DRAWDOWN_CIRCUIT_BREAKER", 0.15),
        # Berhenti entry jika kalah 5 kali berturut-turut
        max_consecutive_losses=_env_int("MAX_CONSECUTIVE_LOSSES", 5),
        max_spread_points=_env_int("MAX_SPREAD_POINTS", 30),
        deviation=_env_int("DEVIATION", 20),
        magic_number=_env_int("MAGIC_NUMBER", 20260311),
        poll_seconds=_env_int("POLL_SECONDS", 30),
        live_sleep_seconds=_env_int("LIVE_SLEEP_SECONDS", 10),
        # Ollama LLM
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_news_model=os.getenv("OLLAMA_NEWS_MODEL", "mistral:7b"),
        ollama_tech_model=os.getenv("OLLAMA_TECH_MODEL", "qwen2.5:14b"),
        ollama_decision_model=os.getenv("OLLAMA_DECISION_MODEL", "deepseek-r1:14b"),
        ollama_enabled=_env_bool("OLLAMA_ENABLED", True),
        ollama_timeout=_env_int("OLLAMA_TIMEOUT", 120),
    )
    return apply_risk_profile(settings)


def apply_risk_profile(settings: Settings) -> Settings:
    profile = settings.risk_profile.strip().upper()
    if profile in {"KECIL", "LOW", "SMALL"}:
        return replace(
            settings,
            risk_profile="KECIL",
            risk_per_trade=max(0.001, settings.risk_per_trade * 0.5),   # Risiko 0.5% per trade
            fixed_lot=max(0.01, settings.fixed_lot),
            prediction_buy_threshold=max(settings.prediction_buy_threshold, 0.63),  # Sinyal sangat ketat
            prediction_sell_threshold=min(settings.prediction_sell_threshold, 0.37),
            max_spread_points=max(5, int(settings.max_spread_points * 0.7)),  # Spread lebih strict
            pair_max_volume_min=min(settings.pair_max_volume_min, 0.05),
            use_trailing_stop=True,
            trailing_stop_pips=max(5.0, settings.trailing_stop_pips * 0.8),
            max_drawdown_circuit_breaker=min(settings.max_drawdown_circuit_breaker, 0.10),  # Henti di 10%
            max_consecutive_losses=min(settings.max_consecutive_losses, 3),
        )
    if profile in {"BESAR", "HIGH", "LARGE"}:
        return replace(
            settings,
            risk_profile="BESAR",
            risk_per_trade=min(0.03, settings.risk_per_trade * 1.5),   # Maks risiko 3% per trade
            fixed_lot=settings.fixed_lot * 1.5,
            prediction_buy_threshold=min(settings.prediction_buy_threshold, 0.55),
            prediction_sell_threshold=max(settings.prediction_sell_threshold, 0.45),
            max_spread_points=max(settings.max_spread_points, 50),
            pair_max_volume_min=max(settings.pair_max_volume_min, 1.0),
            max_drawdown_circuit_breaker=min(0.25, settings.max_drawdown_circuit_breaker * 1.5),
            max_consecutive_losses=min(10, settings.max_consecutive_losses * 2),
        )
    # Profile SEDANG – hormati threshold yang sudah di-set di .env
    # Tidak paksa 0.60/0.40 agar user bisa tuning sendiri
    return replace(settings, risk_profile="SEDANG")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
