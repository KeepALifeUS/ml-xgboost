"""
Technical Indicators for XGBoost Feature Engineering
===================================================

Advanced technical indicators specifically optimized for tree-based models
and time series forecasting. Implements both classical and modern technical
analysis indicators with enterprise patterns.

Indicators Categories:
- Trend Indicators: SMA, EMA, MACD, ADX, etc.
- Momentum Oscillators: RSI, Stochastic, Williams %R
- Volume Indicators: OBV, VWAP, CMF
- Volatility Indicators: Bollinger Bands, ATR, Keltner Channels
- Support/Resistance: Pivot Points, Fibonacci Retracements

Author: ML XGBoost Contributors
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    import warnings
    warnings.warn("TA-Lib not available. Some indicators will use pandas_ta fallback.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import pearsonr, spearmanr
import numba
from numba import jit, prange
from loguru import logger

from .feature_engineer import BaseFeatureGenerator, FeatureType


class IndicatorType(Enum):
    """Типы технических индикаторов"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    STATISTICAL = "statistical"


@dataclass
class TechnicalConfig:
    """Конфигурация технических индикаторов"""
    
    # Периоды для индикаторов
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21, 30])
    atr_periods: List[int] = field(default_factory=lambda: [14, 20, 30])
    
    # MACD параметры
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    
    # ADX
    adx_period: int = 14
    
    # Williams %R
    williams_period: int = 14
    
    # CCI
    cci_period: int = 20
    
    # Включение/выключение групп индикаторов
    enable_trend: bool = True
    enable_momentum: bool = True
    enable_volume: bool = True
    enable_volatility: bool = True
    enable_support_resistance: bool = True
    enable_patterns: bool = True
    
    # Дополнительные опции
    normalize_indicators: bool = True
    add_signal_features: bool = True  # Добавлять buy/sell сигналы
    add_divergence_features: bool = True
    lookback_divergence: int = 20


class TechnicalIndicators(BaseFeatureGenerator):
    """
    Генератор технических индикаторов для tree моделей
    
    Оптимизирован для:
    - XGBoost и tree-based алгоритмов
    - Криптовалютные рынки
    - Высокочастотные данные
    - Нестационарные временные ряды
    """
    
    def __init__(self, feature_config, technical_config: Optional[TechnicalConfig] = None):
        self.feature_config = feature_config
        self.config = technical_config or TechnicalConfig()
        
        # Метаданные созданных признаков
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}
        
        # Кэш для ускорения повторных вычислений
        self._indicator_cache: Dict[str, pd.Series] = {}
        
        logger.info("🔧 Инициализирован генератор технических индикаторов")
    
    def generate_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Генерация технических индикаторов
        
        Args:
            data: OHLCV данные
            target: Целевая переменная (опционально)
        """
        
        start_time = time.time()
        logger.info("📈 Генерация технических индикаторов...")
        
        # Валидация входных данных
        required_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in required_columns if col in data.columns]
        
        if len(available_columns) < 4:
            logger.warning("⚠️ Недостаточно OHLC данных для технических индикаторов")
            return pd.DataFrame(index=data.index)
        
        # Извлечение OHLCV
        open_prices = data['open']
        high_prices = data['high'] 
        low_prices = data['low']
        close_prices = data['close']
        volume = data.get('volume', pd.Series(index=data.index, data=1.0))
        
        # Контейнер для всех индикаторов
        all_indicators = []
        
        # Trend indicators
        if self.config.enable_trend:
            trend_indicators = self._generate_trend_indicators(
                open_prices, high_prices, low_prices, close_prices, volume
            )
            all_indicators.append(trend_indicators)
        
        # Momentum indicators
        if self.config.enable_momentum:
            momentum_indicators = self._generate_momentum_indicators(
                open_prices, high_prices, low_prices, close_prices, volume
            )
            all_indicators.append(momentum_indicators)
        
        # Volume indicators
        if self.config.enable_volume and 'volume' in data.columns:
            volume_indicators = self._generate_volume_indicators(
                open_prices, high_prices, low_prices, close_prices, volume
            )
            all_indicators.append(volume_indicators)
        
        # Volatility indicators
        if self.config.enable_volatility:
            volatility_indicators = self._generate_volatility_indicators(
                open_prices, high_prices, low_prices, close_prices, volume
            )
            all_indicators.append(volatility_indicators)
        
        # Support/Resistance indicators
        if self.config.enable_support_resistance:
            sr_indicators = self._generate_support_resistance_indicators(
                open_prices, high_prices, low_prices, close_prices, volume
            )
            all_indicators.append(sr_indicators)
        
        # Pattern indicators
        if self.config.enable_patterns:
            pattern_indicators = self._generate_pattern_indicators(
                open_prices, high_prices, low_prices, close_prices, volume
            )
            all_indicators.append(pattern_indicators)
        
        # Объединение всех индикаторов
        if all_indicators:
            result_df = pd.concat([df for df in all_indicators if not df.empty], axis=1)
        else:
            result_df = pd.DataFrame(index=data.index)
        
        # Нормализация если включена
        if self.config.normalize_indicators:
            result_df = self._normalize_indicators(result_df)
        
        # Сигнальные признаки
        if self.config.add_signal_features:
            signal_features = self._generate_signal_features(result_df, close_prices)
            result_df = pd.concat([result_df, signal_features], axis=1)
        
        # Дивергенция признаки
        if self.config.add_divergence_features and target is not None:
            divergence_features = self._generate_divergence_features(
                result_df, close_prices, target
            )
            result_df = pd.concat([result_df, divergence_features], axis=1)
        
        # Обновление метаданных
        self.feature_names_ = list(result_df.columns)
        for col in result_df.columns:
            self.feature_types_[col] = FeatureType.NUMERICAL.value
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Создано {len(result_df.columns)} технических индикаторов за {generation_time:.2f}с")
        
        return result_df
    
    def _generate_trend_indicators(
        self, 
        open_prices: pd.Series,
        high_prices: pd.Series, 
        low_prices: pd.Series,
        close_prices: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Генерация трендовых индикаторов"""
        
        indicators = {}
        
        # Simple Moving Averages
        for period in self.config.sma_periods:
            sma = self._calculate_sma(close_prices, period)
            indicators[f'sma_{period}'] = sma
            indicators[f'sma_{period}_ratio'] = close_prices / sma - 1
            indicators[f'sma_{period}_distance'] = (close_prices - sma) / sma
        
        # Exponential Moving Averages
        for period in self.config.ema_periods:
            ema = self._calculate_ema(close_prices, period)
            indicators[f'ema_{period}'] = ema
            indicators[f'ema_{period}_ratio'] = close_prices / ema - 1
        
        # MACD
        macd_line, macd_signal, macd_hist = self._calculate_macd(
            close_prices, self.config.macd_fast, 
            self.config.macd_slow, self.config.macd_signal
        )
        indicators['macd'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        indicators['macd_crossover'] = (macd_line > macd_signal).astype(int)
        
        # ADX (Average Directional Index)
        adx, di_plus, di_minus = self._calculate_adx(
            high_prices, low_prices, close_prices, self.config.adx_period
        )
        indicators['adx'] = adx
        indicators['di_plus'] = di_plus
        indicators['di_minus'] = di_minus
        indicators['di_diff'] = di_plus - di_minus
        
        # Parabolic SAR
        sar = self._calculate_parabolic_sar(high_prices, low_prices)
        indicators['parabolic_sar'] = sar
        indicators['sar_trend'] = (close_prices > sar).astype(int)
        
        # Trend strength
        indicators['trend_strength'] = self._calculate_trend_strength(close_prices)
        
        # Moving average crossovers
        if len(self.config.sma_periods) >= 2:
            fast_ma = self._calculate_sma(close_prices, self.config.sma_periods[0])
            slow_ma = self._calculate_sma(close_prices, self.config.sma_periods[1])
            indicators['ma_crossover'] = (fast_ma > slow_ma).astype(int)
        
        return pd.DataFrame(indicators, index=close_prices.index)
    
    def _generate_momentum_indicators(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series, 
        close_prices: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Генерация моментум индикаторов"""
        
        indicators = {}
        
        # RSI (Relative Strength Index)
        for period in self.config.rsi_periods:
            rsi = self._calculate_rsi(close_prices, period)
            indicators[f'rsi_{period}'] = rsi
            indicators[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
            indicators[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(
            high_prices, low_prices, close_prices,
            self.config.stoch_k, self.config.stoch_d
        )
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        indicators['stoch_crossover'] = (stoch_k > stoch_d).astype(int)
        
        # Williams %R
        williams_r = self._calculate_williams_r(
            high_prices, low_prices, close_prices, self.config.williams_period
        )
        indicators['williams_r'] = williams_r
        
        # CCI (Commodity Channel Index)
        cci = self._calculate_cci(
            high_prices, low_prices, close_prices, self.config.cci_period
        )
        indicators['cci'] = cci
        indicators['cci_overbought'] = (cci > 100).astype(int)
        indicators['cci_oversold'] = (cci < -100).astype(int)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            roc = self._calculate_roc(close_prices, period)
            indicators[f'roc_{period}'] = roc
        
        # Momentum
        for period in [5, 10, 20]:
            momentum = close_prices / close_prices.shift(period) - 1
            indicators[f'momentum_{period}'] = momentum
        
        # Money Flow Index
        if 'volume' in locals():
            mfi = self._calculate_mfi(high_prices, low_prices, close_prices, volume, 14)
            indicators['mfi'] = mfi
        
        return pd.DataFrame(indicators, index=close_prices.index)
    
    def _generate_volume_indicators(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Генерация объемных индикаторов"""
        
        indicators = {}
        
        # On-Balance Volume
        obv = self._calculate_obv(close_prices, volume)
        indicators['obv'] = obv
        indicators['obv_ma'] = obv.rolling(20).mean()
        
        # Volume Price Trend
        vpt = self._calculate_vpt(close_prices, volume)
        indicators['vpt'] = vpt
        
        # Accumulation/Distribution Line
        ad_line = self._calculate_ad_line(high_prices, low_prices, close_prices, volume)
        indicators['ad_line'] = ad_line
        
        # Chaikin Money Flow
        cmf = self._calculate_cmf(high_prices, low_prices, close_prices, volume, 20)
        indicators['cmf'] = cmf
        
        # Volume SMA
        for period in [5, 10, 20]:
            vol_sma = volume.rolling(period).mean()
            indicators[f'volume_sma_{period}'] = vol_sma
            indicators[f'volume_ratio_{period}'] = volume / vol_sma
        
        # VWAP (Volume Weighted Average Price)
        vwap = self._calculate_vwap(high_prices, low_prices, close_prices, volume)
        indicators['vwap'] = vwap
        indicators['vwap_ratio'] = close_prices / vwap - 1
        
        return pd.DataFrame(indicators, index=close_prices.index)
    
    def _generate_volatility_indicators(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Генерация индикаторов волатильности"""
        
        indicators = {}
        
        # Average True Range
        for period in self.config.atr_periods:
            atr = self._calculate_atr(high_prices, low_prices, close_prices, period)
            indicators[f'atr_{period}'] = atr
            indicators[f'atr_{period}_ratio'] = atr / close_prices
        
        # Bollinger Bands
        bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(
            close_prices, self.config.bb_period, self.config.bb_std
        )
        indicators['bb_middle'] = bb_middle
        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        indicators['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        indicators['bb_squeeze'] = (indicators['bb_width'] < indicators['bb_width'].rolling(20).mean()).astype(int)
        
        # Keltner Channels
        kc_middle, kc_upper, kc_lower = self._calculate_keltner_channels(
            high_prices, low_prices, close_prices, 20
        )
        indicators['kc_middle'] = kc_middle
        indicators['kc_upper'] = kc_upper
        indicators['kc_lower'] = kc_lower
        indicators['kc_position'] = (close_prices - kc_lower) / (kc_upper - kc_lower)
        
        # Volatility measures
        for period in [5, 10, 20]:
            returns = close_prices.pct_change()
            volatility = returns.rolling(period).std() * np.sqrt(252)
            indicators[f'volatility_{period}'] = volatility
            
        return pd.DataFrame(indicators, index=close_prices.index)
    
    def _generate_support_resistance_indicators(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Генерация индикаторов поддержки/сопротивления"""
        
        indicators = {}
        
        # Pivot Points
        pivot_points = self._calculate_pivot_points(high_prices, low_prices, close_prices)
        for key, value in pivot_points.items():
            indicators[f'pivot_{key}'] = value
        
        # Fibonacci Retracement levels
        fib_levels = self._calculate_fibonacci_levels(high_prices, low_prices, close_prices)
        for level, value in fib_levels.items():
            indicators[f'fib_{level}'] = value
        
        # Support and resistance levels
        support_levels, resistance_levels = self._calculate_support_resistance(
            high_prices, low_prices, close_prices, window=20
        )
        indicators['support_level'] = support_levels
        indicators['resistance_level'] = resistance_levels
        indicators['support_distance'] = (close_prices - support_levels) / close_prices
        indicators['resistance_distance'] = (resistance_levels - close_prices) / close_prices
        
        return pd.DataFrame(indicators, index=close_prices.index)
    
    def _generate_pattern_indicators(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Генерация паттерн индикаторов"""
        
        indicators = {}
        
        # Candlestick patterns (если доступен TA-Lib)
        if TALIB_AVAILABLE:
            # Doji patterns
            indicators['doji'] = talib.CDLDOJI(open_prices.values, high_prices.values, 
                                            low_prices.values, close_prices.values)
            
            # Hammer patterns
            indicators['hammer'] = talib.CDLHAMMER(open_prices.values, high_prices.values,
                                                 low_prices.values, close_prices.values)
            
            # Engulfing patterns
            indicators['engulfing'] = talib.CDLENGULFING(open_prices.values, high_prices.values,
                                                       low_prices.values, close_prices.values)
            
            # Morning/Evening star
            indicators['morning_star'] = talib.CDLMORNINGSTAR(open_prices.values, high_prices.values,
                                                            low_prices.values, close_prices.values)
            
            indicators['evening_star'] = talib.CDLEVENINGSTAR(open_prices.values, high_prices.values,
                                                            low_prices.values, close_prices.values)
        
        # Price action patterns
        indicators['higher_high'] = self._detect_higher_highs(high_prices)
        indicators['lower_low'] = self._detect_lower_lows(low_prices)
        indicators['double_top'] = self._detect_double_top(high_prices)
        indicators['double_bottom'] = self._detect_double_bottom(low_prices)
        
        # Gap detection
        indicators['gap_up'] = (open_prices > high_prices.shift(1)).astype(int)
        indicators['gap_down'] = (open_prices < low_prices.shift(1)).astype(int)
        
        return pd.DataFrame(indicators, index=close_prices.index)
    
    def _normalize_indicators(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация индикаторов для tree моделей"""
        
        normalized_df = indicators_df.copy()
        
        for col in indicators_df.columns:
            values = indicators_df[col].dropna()
            
            if len(values) == 0:
                continue
            
            # Определение типа нормализации на основе диапазона значений
            min_val, max_val = values.min(), values.max()
            
            if min_val >= 0 and max_val <= 1:
                # Уже нормализованные значения (0-1)
                continue
            elif min_val >= -1 and max_val <= 1:
                # Уже нормализованные значения (-1 to 1)
                continue
            elif 'rsi' in col.lower() or 'stoch' in col.lower():
                # RSI и Stochastic уже в диапазоне 0-100, нормализуем к 0-1
                normalized_df[col] = indicators_df[col] / 100.0
            elif 'williams' in col.lower():
                # Williams %R в диапазоне -100 to 0, нормализуем к 0-1
                normalized_df[col] = (indicators_df[col] + 100) / 100.0
            elif abs(max_val - min_val) > 1000:
                # Большой диапазон - применяем robust scaling
                median_val = values.median()
                mad = (values - median_val).abs().median()
                if mad > 0:
                    normalized_df[col] = (indicators_df[col] - median_val) / (1.4826 * mad)
            else:
                # Стандартная нормализация
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    normalized_df[col] = (indicators_df[col] - mean_val) / std_val
        
        return normalized_df
    
    def _generate_signal_features(
        self,
        indicators_df: pd.DataFrame,
        close_prices: pd.Series
    ) -> pd.DataFrame:
        """Генерация сигнальных признаков"""
        
        signals = {}
        
        # RSI сигналы
        rsi_cols = [col for col in indicators_df.columns if col.startswith('rsi_')]
        for rsi_col in rsi_cols:
            if rsi_col in indicators_df.columns:
                rsi = indicators_df[rsi_col]
                signals[f'{rsi_col}_buy_signal'] = ((rsi < 30) & (rsi.shift(1) >= 30)).astype(int)
                signals[f'{rsi_col}_sell_signal'] = ((rsi > 70) & (rsi.shift(1) <= 70)).astype(int)
        
        # MACD сигналы
        if 'macd' in indicators_df.columns and 'macd_signal' in indicators_df.columns:
            macd = indicators_df['macd']
            macd_signal = indicators_df['macd_signal']
            
            signals['macd_buy_signal'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)
            signals['macd_sell_signal'] = ((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))).astype(int)
        
        # Bollinger Bands сигналы
        if 'bb_upper' in indicators_df.columns and 'bb_lower' in indicators_df.columns:
            bb_upper = indicators_df['bb_upper']
            bb_lower = indicators_df['bb_lower']
            
            signals['bb_buy_signal'] = ((close_prices < bb_lower) & (close_prices.shift(1) >= bb_lower.shift(1))).astype(int)
            signals['bb_sell_signal'] = ((close_prices > bb_upper) & (close_prices.shift(1) <= bb_upper.shift(1))).astype(int)
        
        return pd.DataFrame(signals, index=close_prices.index)
    
    def _generate_divergence_features(
        self,
        indicators_df: pd.DataFrame,
        close_prices: pd.Series,
        target: pd.Series
    ) -> pd.DataFrame:
        """Генерация признаков дивергенции"""
        
        divergences = {}
        lookback = self.config.lookback_divergence
        
        # RSI дивергенция
        rsi_cols = [col for col in indicators_df.columns if col.startswith('rsi_') and not col.endswith('_signal')]
        for rsi_col in rsi_cols:
            if rsi_col in indicators_df.columns:
                rsi = indicators_df[rsi_col]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                price_lower_low = (close_prices < close_prices.shift(lookback)).astype(int)
                rsi_higher_low = (rsi > rsi.shift(lookback)).astype(int)
                divergences[f'{rsi_col}_bullish_div'] = price_lower_low & rsi_higher_low
                
                # Bearish divergence: price makes higher high, RSI makes lower high  
                price_higher_high = (close_prices > close_prices.shift(lookback)).astype(int)
                rsi_lower_high = (rsi < rsi.shift(lookback)).astype(int)
                divergences[f'{rsi_col}_bearish_div'] = price_higher_high & rsi_lower_high
        
        return pd.DataFrame(divergences, index=close_prices.index)
    
    # Technical indicator calculation methods
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int, 
        slow: int, 
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD calculation"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        d_period: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        
        # True Range
        atr = self._calculate_atr(high, low, close, period)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int,
        std_dev: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return sma, upper_band, lower_band
    
    def _calculate_keltner_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        ema = self._calculate_ema(close, period)
        atr = self._calculate_atr(high, low, close, period)
        upper_channel = ema + (2 * atr)
        lower_channel = ema - (2 * atr)
        return ema, upper_channel, lower_channel
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend"""
        vpt = (volume * close.pct_change()).cumsum()
        return vpt
    
    def _calculate_ad_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad_line = (clv * volume).cumsum()
        return ad_line
    
    def _calculate_cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int
    ) -> pd.Series:
        """Chaikin Money Flow"""
        mfv = ((close - low) - (high - close)) / (high - low) * volume
        mfv = mfv.fillna(0)
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    def _calculate_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Rate of Change"""
        roc = prices.pct_change(periods=period) * 100
        return roc
    
    def _calculate_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int
    ) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_parabolic_sar(
        self,
        high: pd.Series,
        low: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> pd.Series:
        """Parabolic SAR"""
        
        # Простая реализация Parabolic SAR
        sar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        af = pd.Series(index=high.index, dtype=float)
        ep = pd.Series(index=high.index, dtype=float)
        
        # Инициализация
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
        af.iloc[0] = acceleration
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(high)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return sar
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Расчет силы тренда"""
        
        # Linear regression slope
        def calculate_slope(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return 0
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        trend_strength = prices.rolling(window=period).apply(calculate_slope)
        return trend_strength
    
    def _calculate_pivot_points(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Расчет пивот поинтов"""
        
        # Daily pivot points
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        
        r1 = 2 * pivot - low.shift(1)
        s1 = 2 * pivot - high.shift(1)
        
        r2 = pivot + (high.shift(1) - low.shift(1))
        s2 = pivot - (high.shift(1) - low.shift(1))
        
        r3 = high.shift(1) + 2 * (pivot - low.shift(1))
        s3 = low.shift(1) - 2 * (high.shift(1) - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def _calculate_fibonacci_levels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> Dict[str, pd.Series]:
        """Расчет уровней Фибоначчи"""
        
        # Swing highs and lows over period
        swing_high = high.rolling(window=period).max()
        swing_low = low.rolling(window=period).min()
        
        diff = swing_high - swing_low
        
        # Fibonacci retracement levels
        fib_levels = {
            '23.6': swing_high - 0.236 * diff,
            '38.2': swing_high - 0.382 * diff,
            '50.0': swing_high - 0.5 * diff,
            '61.8': swing_high - 0.618 * diff,
            '78.6': swing_high - 0.786 * diff,
        }
        
        return fib_levels
    
    def _calculate_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """Расчет уровней поддержки и сопротивления"""
        
        # Rolling support and resistance based on min/max
        support = low.rolling(window=window, center=True).min()
        resistance = high.rolling(window=window, center=True).max()
        
        # Forward fill for recent periods
        support = support.fillna(method='ffill')
        resistance = resistance.fillna(method='ffill')
        
        return support, resistance
    
    def _detect_higher_highs(self, high: pd.Series, window: int = 5) -> pd.Series:
        """Обнаружение растущих максимумов"""
        
        local_maxima = (high.shift(window) < high) & (high.shift(-window) < high)
        higher_highs = local_maxima & (high > high.shift(window*2))
        return higher_highs.astype(int)
    
    def _detect_lower_lows(self, low: pd.Series, window: int = 5) -> pd.Series:
        """Обнаружение падающих минимумов"""
        
        local_minima = (low.shift(window) > low) & (low.shift(-window) > low)
        lower_lows = local_minima & (low < low.shift(window*2))
        return lower_lows.astype(int)
    
    def _detect_double_top(self, high: pd.Series, window: int = 10) -> pd.Series:
        """Обнаружение двойной вершины"""
        
        peaks = find_peaks(high, distance=window)[0]
        double_tops = pd.Series(0, index=high.index)
        
        for i in range(1, len(peaks)):
            peak1_idx, peak2_idx = peaks[i-1], peaks[i]
            peak1_val, peak2_val = high.iloc[peak1_idx], high.iloc[peak2_idx]
            
            # Check if peaks are similar height (within 2%)
            if abs(peak1_val - peak2_val) / max(peak1_val, peak2_val) < 0.02:
                double_tops.iloc[peak2_idx] = 1
        
        return double_tops
    
    def _detect_double_bottom(self, low: pd.Series, window: int = 10) -> pd.Series:
        """Обнаружение двойного дна"""
        
        troughs = find_peaks(-low, distance=window)[0]
        double_bottoms = pd.Series(0, index=low.index)
        
        for i in range(1, len(troughs)):
            trough1_idx, trough2_idx = troughs[i-1], troughs[i]
            trough1_val, trough2_val = low.iloc[trough1_idx], low.iloc[trough2_idx]
            
            # Check if troughs are similar height (within 2%)
            if abs(trough1_val - trough2_val) / max(trough1_val, trough2_val) < 0.02:
                double_bottoms.iloc[trough2_idx] = 1
        
        return double_bottoms
    
    def get_feature_names(self) -> List[str]:
        """Получение имен созданных признаков"""
        return self.feature_names_
    
    def get_feature_types(self) -> Dict[str, str]:
        """Получение типов признаков"""
        return self.feature_types_


# JIT-compiled utility functions для ускорения
@jit(nopython=True)
def fast_ema_calculation(prices: np.ndarray, alpha: float) -> np.ndarray:
    """Быстрое вычисление EMA с Numba"""
    
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


@jit(nopython=True)
def fast_rsi_calculation(prices: np.ndarray, period: int) -> np.ndarray:
    """Быстрое вычисление RSI с Numba"""
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.empty(len(prices))
    rsi[0] = 50  # Neutral starting point
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(prices)):
        if i == period:
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
        
        rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


if __name__ == "__main__":
    # Пример использования
    logger.info("🧪 Тестирование Technical Indicators...")
    
    # Создание синтетических OHLCV данных
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Симуляция реалистичных OHLCV данных
    price_base = 100
    price_walk = np.cumsum(np.random.randn(n_samples) * 0.001) + price_base
    
    ohlcv_data = pd.DataFrame(index=dates)
    ohlcv_data['close'] = price_walk
    ohlcv_data['open'] = ohlcv_data['close'].shift(1).fillna(price_base)
    ohlcv_data['high'] = ohlcv_data[['open', 'close']].max(axis=1) + np.random.exponential(0.1, n_samples)
    ohlcv_data['low'] = ohlcv_data[['open', 'close']].min(axis=1) - np.random.exponential(0.1, n_samples)
    ohlcv_data['volume'] = np.random.exponential(1000, n_samples)
    
    # Создание генератора индикаторов
    config = TechnicalConfig(
        sma_periods=[5, 10, 20],
        ema_periods=[5, 10, 20],
        rsi_periods=[14, 21],
        enable_patterns=False  # Отключаем для быстроты тестирования
    )
    
    # Mock feature config для совместимости
    from types import SimpleNamespace
    mock_feature_config = SimpleNamespace()
    
    generator = TechnicalIndicators(mock_feature_config, config)
    
    # Генерация индикаторов
    start_time = time.time()
    technical_features = generator.generate_features(ohlcv_data)
    generation_time = time.time() - start_time
    
    logger.info(f"📊 Создано {len(technical_features.columns)} технических индикаторов")
    logger.info(f"⏱️ Время генерации: {generation_time:.2f}с")
    logger.info(f"🎯 Индикаторы: {technical_features.columns.tolist()[:10]}...")  # Первые 10
    
    # Проверка на пропущенные значения
    missing_pct = technical_features.isnull().sum() / len(technical_features) * 100
    high_missing = missing_pct[missing_pct > 50]
    
    if not high_missing.empty:
        logger.warning(f"⚠️ Высокий процент пропусков в: {high_missing.index.tolist()}")
    else:
        logger.info("✅ Качество данных хорошее (< 50% пропусков)")
    
    logger.info("✅ Тестирование Technical Indicators завершено!")