import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict

from trading_bot.config import Settings
from trading_bot.modeling import train_model, predict_proba, save_artifacts
from trading_bot.data_pipeline import make_sequences
from trading_bot.workflows import split_train_val

class StrategistAI:
    """
    Strategist Agent (The Brain).
    Responsible for analyzing price data and generating probabilities using Deep Learning.
    Implements Transfer Learning by sharing weights across different market pairs.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def analyze_and_predict(self, symbol: str, frame: pd.DataFrame, feature_cols: list[str], train_ratio: float = 0.7) -> Tuple[np.ndarray, pd.Index]:
        """
        Trains the global model on the provided pair data (updating shared weights),
        then generates signal probabilities for the unseen validation data.
        """
        try:
            dataset, scaler = make_sequences(
                frame=frame,
                feature_cols=feature_cols,
                lookback=self.settings.lookback,
                scaler=None,
                fit_scaler=True,
            )
            
            if len(dataset.x) < 500:
                print(f"[Strategist] Not enough samples ({len(dataset.x)} < 500) for {symbol}. Skipping training.")
                return np.array([]), pd.Index([])

            split = split_train_val(dataset.x, dataset.y, train_ratio=train_ratio)
            
            if len(split.x_val) == 0:
                print(f"[Strategist] Validation set empty for {symbol}. Skipping training.")
                return np.array([]), pd.Index([])

            print(f"[Strategist] Initiating Deep Learning for {symbol} on {len(split.x_train)} samples...")
            model, _ = train_model(
                x_train=split.x_train,
                y_train=split.y_train,
                x_val=split.x_val,
                y_val=split.y_val,
                settings=self.settings,
            )
            
            print(f"[Strategist] Generating predictions for {symbol}...")
            # Predict only on validation set for backtesting to avoid data leakage
            probs = predict_proba(model, split.x_val)
            timestamps = dataset.timestamps[len(split.x_train) :]
            
            # Update physical artifacts (meta, scaler) so live demo can load them
            save_artifacts(model, scaler, feature_cols, self.settings.lookback)
            
            return probs, timestamps
            
        except Exception as e:
            print(f"[Strategist] Error analyzing {symbol}: {e}")
            return np.array([]), pd.Index([])
