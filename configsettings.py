"""
Central configuration management for the trading engine.
Uses Pydantic for type-safe settings with environment variable fallbacks.
"""
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client
import logging

class AssetClass(Enum):
    """Supported asset classes"""
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    FOREX = "forex"

class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    ALPACA = "alpaca"  # For stocks
    OANDA = "oanda"    # For forex

@dataclass
class DataConfig:
    """Data collection configuration"""
    historical_days: int = 365
    timeframe: str = "1h"  # 1 hour candles
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: int = 5
    
    # Asset-specific settings
    crypto_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"
    ])
    stock_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL"
    ])
    commodity_symbols: List[str] = field(default_factory=lambda: [
        "GC=F",  # Gold
        "CL=F",  # Crude Oil
        "SI=F",  # Silver
    ])

@dataclass
class ModelConfig:
    """ML model configuration"""
    lookback_window: int = 100
    prediction_horizon: int = 24  # Predict 24 hours ahead
    train_test_split: float = 0.8
    validation_split: float = 0.1
    feature_count: int = 50
    
    # Ensemble models
    models_to_train: List[str] = field(default_factory=lambda: [
        "xgboost",
        "lightgbm",
        "random_forest",
        "lstm"
    ])
    
    # Hyperparameter tuning
    n_trials: int = 50
    cv_folds: int = 3

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% per trade
    max_portfolio_risk: float = 0.02  # 2% max loss
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_daily_trades: int = 10
    max_leverage: float = 3.0
    
    # Correlation limits
    max_correlation_threshold: float = 0.7
    min_diversification_score: float = 0.5

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    collection_prefix: str = "trading_engine"
    
    # Collections
    collections: Dict[str, str] = field(default_factory=lambda: {
        "market_data": "market_data",
        "predictions": "predictions",
        "trades": "executed_trades",
        "performance": "performance_metrics",
        "models": "trained_models",
        "errors": "system_errors"
    })

class TradingEngineConfig:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.risk = RiskConfig()
        self.firebase = FirebaseConfig()