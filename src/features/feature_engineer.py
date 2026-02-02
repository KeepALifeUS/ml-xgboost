"""
Advanced Feature Engineering for XGBoost Tree Models
====================================================

Comprehensive feature engineering pipeline specifically optimized for tree-based models
and time series data. Implements enterprise patterns for
production-ready feature generation.

Key Features:
- Tree-optimized feature transformations
- Time Series-specific indicators
- Time series feature engineering
- Automated feature selection
- Feature interaction detection
- Performance optimization with caching

Author: ML XGBoost Contributors
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import pickle
import joblib
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    KBinsDiscretizer, LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_regression, f_classif,
    mutual_info_regression, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.decomposition import PCA, FastICA
from sklearn.impute import SimpleImputer, KNNImputer
import pandas_ta as ta
import talib
import scipy.stats as stats
from scipy.signal import find_peaks
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numba
from numba import jit, prange


class FeatureType(Enum):
    """Типы признаков для оптимизации под tree модели"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    TEMPORAL = "temporal"
    INTERACTION = "interaction"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    LOG_TRANSFORM = "log_transform"
    BINNED = "binned"


class ScalingMethod(Enum):
    """Методы масштабирования данных"""
    NONE = "none"
    STANDARD = "standard"  # Z-score normalization
    ROBUST = "robust"  # Median and IQR
    MINMAX = "minmax"  # Min-max scaling
    QUANTILE = "quantile"  # Quantile transformation
    LOG = "log"  # Log transformation
    SQRT = "sqrt"  # Square root transformation


@dataclass
class FeatureConfig:
    """Конфигурация feature engineering"""
    
    # Основные параметры
    enable_technical_indicators: bool = True
    enable_statistical_features: bool = True
    enable_lag_features: bool = True
    enable_interaction_features: bool = True
    enable_volatility_features: bool = True
    
    # Technical indicators
    technical_periods: List[int] = field(default_factory=lambda: [5, 10, 14, 20, 30, 50])
    enable_oscillators: bool = True
    enable_trend_indicators: bool = True
    enable_volume_indicators: bool = True
    
    # Statistical features
    statistical_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    enable_moments: bool = True  # skewness, kurtosis
    enable_quantiles: bool = True
    enable_entropy: bool = True
    
    # Lag features
    max_lags: int = 20
    lag_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 20])
    enable_rolling_lags: bool = True
    
    # Interaction features
    max_interaction_degree: int = 2
    interaction_threshold: float = 0.1  # Minimum correlation for interaction
    auto_detect_interactions: bool = True
    
    # Volatility features  
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30])
    enable_garch_features: bool = True
    enable_realized_volatility: bool = True
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # mutual_info, f_score, variance
    feature_selection_k: int = 100  # Top K features
    variance_threshold: float = 0.01
    
    # Data preprocessing
    handle_missing: str = "forward_fill"  # forward_fill, interpolate, knn, drop
    scaling_method: ScalingMethod = ScalingMethod.ROBUST
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Performance optimization
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    
    # Categorical features
    categorical_encoding: str = "target"  # target, onehot, label, binary
    max_cardinality: int = 50
    enable_binning: bool = True
    binning_strategy: str = "quantile"  # quantile, uniform, kmeans


@dataclass
class FeatureMetadata:
    """Метаданные о созданных признаках"""
    
    feature_names: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    creation_time: Dict[str, float] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    selected_features: List[str] = field(default_factory=list)
    removed_features: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'feature_names': self.feature_names,
            'feature_types': self.feature_types,
            'feature_importance': self.feature_importance,
            'creation_time': self.creation_time,
            'feature_stats': self.feature_stats,
            'selected_features': self.selected_features,
            'removed_features': self.removed_features,
        }


class BaseFeatureGenerator(ABC):
    """Базовый класс для генераторов признаков - enterprise pattern"""
    
    @abstractmethod
    def generate_features(
        self, 
        data: pd.DataFrame, 
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Генерация признаков"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Получение имен созданных признаков"""
        pass
    
    @abstractmethod
    def get_feature_types(self) -> Dict[str, str]:
        """Получение типов признаков"""
        pass


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Продвинутый Feature Engineer для XGBoost моделей
    
    Реализует enterprise patterns:
    - Модульная архитектура с подключаемыми генераторами
    - Кэширование для производительности
    - Параллельная обработка данных
    - Валидация и мониторинг качества данных
    - Автоматическая оптимизация для tree моделей
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Состояние объекта
        self.feature_generators_: List[BaseFeatureGenerator] = []
        self.scalers_: Dict[str, Any] = {}
        self.feature_selector_: Optional[Any] = None
        self.metadata_: FeatureMetadata = FeatureMetadata()
        self.is_fitted_: bool = False
        
        # Кэширование
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        
        self.console = Console()
        self._setup_logging()
        self._initialize_generators()
    
    def _setup_logging(self):
        """Настройка логирования"""
        logger.add(
            f"logs/feature_engineer_{datetime.now():%Y%m%d}.log",
            rotation="daily",
            retention="30 days", 
            level="INFO"
        )
    
    def _initialize_generators(self):
        """Инициализация генераторов признаков"""
        
        logger.info("🔧 Инициализация генераторов признаков...")
        
        # Импортируем генераторы
        from .technical_indicators import TechnicalIndicators
        from .statistical_features import StatisticalFeatures
        from .lagging_features import LaggingFeatures
        from .interaction_features import InteractionFeatures
        from .volatility_features import VolatilityFeatures
        
        # Создаем генераторы на основе конфигурации
        if self.config.enable_technical_indicators:
            self.feature_generators_.append(TechnicalIndicators(self.config))
        
        if self.config.enable_statistical_features:
            self.feature_generators_.append(StatisticalFeatures(self.config))
        
        if self.config.enable_lag_features:
            self.feature_generators_.append(LaggingFeatures(self.config))
        
        if self.config.enable_interaction_features:
            self.feature_generators_.append(InteractionFeatures(self.config))
        
        if self.config.enable_volatility_features:
            self.feature_generators_.append(VolatilityFeatures(self.config))
        
        logger.info(f"✅ Инициализировано {len(self.feature_generators_)} генераторов")
    
    def _validate_input_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Валидация и очистка входных данных"""
        
        logger.info("🔍 Валидация входных данных...")
        
        X = X.copy()
        
        # Проверка на пустые данные
        if X.empty:
            raise ValueError("Входные данные пусты")
        
        # Проверка на дублированные колонки
        if X.columns.duplicated().any():
            logger.warning("⚠️ Найдены дублированные колонки, удаляем...")
            X = X.loc[:, ~X.columns.duplicated()]
        
        # Обработка пропущенных значений
        missing_ratio = X.isnull().sum() / len(X)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        
        if high_missing_cols:
            logger.warning(f"⚠️ Удаляем колонки с >50% пропусков: {high_missing_cols}")
            X = X.drop(columns=high_missing_cols)
        
        # Обработка оставшихся пропусков
        if self.config.handle_missing == "forward_fill":
            X = X.fillna(method='ffill').fillna(method='bfill')
        elif self.config.handle_missing == "interpolate":
            X = X.interpolate().fillna(method='bfill')
        elif self.config.handle_missing == "drop":
            X = X.dropna()
        
        # Обработка inf значений
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"✅ Данные валидированы: {X.shape}")
        
        return X
    
    def _detect_feature_types(self, X: pd.DataFrame) -> Dict[str, str]:
        """Автоматическое определение типов признаков"""
        
        feature_types = {}
        
        for col in X.columns:
            dtype = X[col].dtype
            unique_values = X[col].nunique()
            unique_ratio = unique_values / len(X)
            
            if dtype in ['object', 'category']:
                if unique_values <= self.config.max_cardinality:
                    feature_types[col] = FeatureType.CATEGORICAL.value
                else:
                    feature_types[col] = FeatureType.CATEGORICAL.value  # Large cardinality
            elif dtype == 'bool':
                feature_types[col] = FeatureType.BINARY.value
            elif unique_values == 2:
                feature_types[col] = FeatureType.BINARY.value
            elif unique_values < 10 and unique_ratio < 0.1:
                feature_types[col] = FeatureType.ORDINAL.value
            else:
                feature_types[col] = FeatureType.NUMERICAL.value
        
        return feature_types
    
    def _remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Удаление выбросов"""
        
        if not self.config.remove_outliers:
            return X
        
        logger.info(f"🧹 Удаление выбросов методом {self.config.outlier_method}...")
        
        X_clean = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if self.config.outlier_method == "iqr":
            for col in numerical_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clipping instead of removal to preserve data
                X_clean[col] = X[col].clip(lower_bound, upper_bound)
        
        elif self.config.outlier_method == "zscore":
            for col in numerical_cols:
                z_scores = np.abs(stats.zscore(X[col]))
                # Clip values beyond 3 standard deviations
                threshold_indices = z_scores > 3
                if threshold_indices.any():
                    median_val = X[col].median()
                    X_clean.loc[threshold_indices, col] = median_val
        
        logger.info("✅ Выбросы обработаны")
        
        return X_clean
    
    def _create_base_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Создание базовых признаков всеми генераторами"""
        
        logger.info("🏗️ Создание признаков...")
        
        all_features = [X]  # Начинаем с оригинальных признаков
        
        if self.config.parallel_processing:
            # Параллельная генерация признаков
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                
                future_to_generator = {
                    executor.submit(generator.generate_features, X, y): generator
                    for generator in self.feature_generators_
                }
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                ) as progress:
                    
                    task = progress.add_task("Генерация признаков...", total=len(self.feature_generators_))
                    
                    for future in as_completed(future_to_generator):
                        generator = future_to_generator[future]
                        
                        try:
                            features = future.result()
                            if features is not None and not features.empty:
                                all_features.append(features)
                                
                                # Обновляем метаданные
                                gen_feature_names = generator.get_feature_names()
                                gen_feature_types = generator.get_feature_types()
                                
                                self.metadata_.feature_names.extend(gen_feature_names)
                                self.metadata_.feature_types.update(gen_feature_types)
                                
                            logger.info(f"✅ {generator.__class__.__name__}: {len(features.columns) if features is not None else 0} признаков")
                            
                        except Exception as e:
                            logger.error(f"❌ Ошибка в {generator.__class__.__name__}: {e}")
                        
                        progress.advance(task)
        else:
            # Последовательная генерация
            for generator in self.feature_generators_:
                try:
                    features = generator.generate_features(X, y)
                    if features is not None and not features.empty:
                        all_features.append(features)
                        
                        gen_feature_names = generator.get_feature_names()
                        gen_feature_types = generator.get_feature_types()
                        
                        self.metadata_.feature_names.extend(gen_feature_names)
                        self.metadata_.feature_types.update(gen_feature_types)
                        
                    logger.info(f"✅ {generator.__class__.__name__}: {len(features.columns) if features is not None else 0} признаков")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка в {generator.__class__.__name__}: {e}")
        
        # Объединяем все признаки
        if len(all_features) > 1:
            X_features = pd.concat(all_features, axis=1)
            
            # Удаляем дублированные колонки
            X_features = X_features.loc[:, ~X_features.columns.duplicated()]
        else:
            X_features = X
        
        logger.info(f"🎯 Создано {len(X_features.columns)} признаков (было {len(X.columns)})")
        
        return X_features
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применение масштабирования данных"""
        
        if self.config.scaling_method == ScalingMethod.NONE:
            return X
        
        logger.info(f"📏 Применение масштабирования: {self.config.scaling_method.value}")
        
        X_scaled = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in self.scalers_:
                if self.config.scaling_method == ScalingMethod.STANDARD:
                    scaler = StandardScaler()
                elif self.config.scaling_method == ScalingMethod.ROBUST:
                    scaler = RobustScaler()
                elif self.config.scaling_method == ScalingMethod.MINMAX:
                    scaler = MinMaxScaler()
                elif self.config.scaling_method == ScalingMethod.QUANTILE:
                    scaler = QuantileTransformer()
                elif self.config.scaling_method == ScalingMethod.LOG:
                    # Log transform (add 1 to handle zeros)
                    X_scaled[col] = np.log1p(X[col] - X[col].min() + 1)
                    continue
                elif self.config.scaling_method == ScalingMethod.SQRT:
                    # Square root transform
                    X_scaled[col] = np.sqrt(X[col] - X[col].min() + 1)
                    continue
                
                self.scalers_[col] = scaler
                X_scaled[col] = scaler.fit_transform(X[[col]]).flatten()
            else:
                X_scaled[col] = self.scalers_[col].transform(X[[col]]).flatten()
        
        return X_scaled
    
    def _apply_binning(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применение binning для categorical features"""
        
        if not self.config.enable_binning:
            return X
        
        logger.info("🗂️ Применение binning для categorical features...")
        
        X_binned = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Создаем binned версии numerical features
            try:
                if self.config.binning_strategy == "quantile":
                    X_binned[f'{col}_binned'] = pd.qcut(
                        X[col], q=10, labels=False, duplicates='drop'
                    )
                elif self.config.binning_strategy == "uniform":
                    X_binned[f'{col}_binned'] = pd.cut(
                        X[col], bins=10, labels=False, duplicates='drop'
                    )
                
            except Exception as e:
                logger.warning(f"⚠️ Не удалось применить binning к {col}: {e}")
        
        return X_binned
    
    def _select_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Отбор наиболее важных признаков"""
        
        if not self.config.enable_feature_selection or y is None:
            return X
        
        logger.info(f"🎯 Отбор признаков методом {self.config.feature_selection_method}...")
        
        # Удаление признаков с низкой дисперсией
        variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_variance_selected = pd.DataFrame(
            variance_selector.fit_transform(X),
            columns=X.columns[variance_selector.get_support()],
            index=X.index
        )
        
        removed_by_variance = set(X.columns) - set(X_variance_selected.columns)
        if removed_by_variance:
            logger.info(f"🗑️ Удалено по variance threshold: {len(removed_by_variance)} признаков")
            self.metadata_.removed_features.extend(list(removed_by_variance))
        
        # Основной feature selection
        if self.config.feature_selection_method == "mutual_info":
            if y.dtype == 'object' or y.nunique() < 10:
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
            else:
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
        elif self.config.feature_selection_method == "f_score":
            if y.dtype == 'object' or y.nunique() < 10:
                selector = SelectKBest(
                    score_func=f_classif,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
            else:
                selector = SelectKBest(
                    score_func=f_regression,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
        else:
            return X_variance_selected
        
        self.feature_selector_ = selector
        X_selected = pd.DataFrame(
            selector.fit_transform(X_variance_selected, y),
            columns=X_variance_selected.columns[selector.get_support()],
            index=X.index
        )
        
        # Обновляем метаданные
        selected_features = list(X_selected.columns)
        removed_features = set(X_variance_selected.columns) - set(selected_features)
        
        self.metadata_.selected_features = selected_features
        self.metadata_.removed_features.extend(list(removed_features))
        
        # Сохраняем важность признаков
        if hasattr(selector, 'scores_'):
            for i, feature in enumerate(X_variance_selected.columns):
                if selector.get_support()[i]:
                    self.metadata_.feature_importance[feature] = selector.scores_[i]
        
        logger.info(f"✅ Отобрано {len(selected_features)} из {len(X.columns)} признаков")
        
        return X_selected
    
    def _calculate_feature_stats(self, X: pd.DataFrame):
        """Вычисление статистик по признакам"""
        
        logger.info("📊 Вычисление статистик признаков...")
        
        for col in X.columns:
            stats_dict = {
                'dtype': str(X[col].dtype),
                'missing_ratio': X[col].isnull().sum() / len(X),
                'unique_values': X[col].nunique(),
                'unique_ratio': X[col].nunique() / len(X),
            }
            
            if X[col].dtype in ['int64', 'float64']:
                stats_dict.update({
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'q25': X[col].quantile(0.25),
                    'median': X[col].median(),
                    'q75': X[col].quantile(0.75),
                    'skewness': X[col].skew(),
                    'kurtosis': X[col].kurtosis(),
                })
            
            self.metadata_.feature_stats[col] = stats_dict
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'FeatureEngineer':
        """
        Обучение feature engineer
        
        Args:
            X: Исходные данные
            y: Целевая переменная (опционально)
        """
        
        start_time = time.time()
        logger.info("🚀 Начало обучения Feature Engineer...")
        
        # Конвертация в DataFrame если нужно
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        
        # Валидация данных
        X = self._validate_input_data(X)
        
        # Определение типов признаков
        base_feature_types = self._detect_feature_types(X)
        self.metadata_.feature_types.update(base_feature_types)
        
        # Удаление выбросов
        X = self._remove_outliers(X)
        
        # Создание признаков
        X_features = self._create_base_features(X, y)
        
        # Применение binning
        X_features = self._apply_binning(X_features)
        
        # Масштабирование
        X_features = self._apply_scaling(X_features)
        
        # Отбор признаков
        X_final = self._select_features(X_features, y)
        
        # Вычисление статистик
        self._calculate_feature_stats(X_final)
        
        # Корреляционная матрица
        if len(X_final.columns) <= 100:  # Ограничиваем для производительности
            self.metadata_.correlation_matrix = X_final.corr()
        
        self.is_fitted_ = True
        
        training_time = time.time() - start_time
        logger.info(f"✅ Feature Engineer обучен за {training_time:.2f}с")
        logger.info(f"📊 Итого признаков: {len(X_final.columns)}")
        
        self._display_summary()
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение feature engineering к новым данным
        
        Args:
            X: Новые данные
            
        Returns:
            Трансформированные данные
        """
        
        if not self.is_fitted_:
            raise ValueError("Feature Engineer не обучен. Вызовите fit() сначала.")
        
        # Конвертация в DataFrame если нужно
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Валидация данных
        X = self._validate_input_data(X)
        
        # Удаление выбросов
        X = self._remove_outliers(X)
        
        # Создание признаков (без target)
        X_features = self._create_base_features(X, y=None)
        
        # Применение binning
        X_features = self._apply_binning(X_features)
        
        # Масштабирование
        X_features = self._apply_scaling(X_features)
        
        # Отбор признаков (применяем уже обученный selector)
        if self.feature_selector_ is not None:
            # Берем только те колонки, которые были в обучении
            available_features = [col for col in self.metadata_.selected_features if col in X_features.columns]
            X_final = X_features[available_features]
        else:
            X_final = X_features
        
        return X_final
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Обучение и трансформация за один вызов"""
        
        self.fit(X, y)
        return self.transform(X)
    
    def _display_summary(self):
        """Отображение summary результатов"""
        
        table = Table(title="🎯 FEATURE ENGINEERING SUMMARY")
        
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Всего признаков", str(len(self.metadata_.feature_names)))
        table.add_row("Отобрано признаков", str(len(self.metadata_.selected_features)))
        table.add_row("Удалено признаков", str(len(self.metadata_.removed_features)))
        table.add_row("Генераторов активно", str(len(self.feature_generators_)))
        
        # Статистика по типам признаков
        type_counts = {}
        for feature_type in self.metadata_.feature_types.values():
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        
        for ftype, count in type_counts.items():
            table.add_row(f"Тип {ftype}", str(count))
        
        self.console.print(table)
    
    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """Получение важности признаков"""
        
        if not self.metadata_.feature_importance:
            logger.warning("⚠️ Важность признаков не вычислена")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in self.metadata_.feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        if top_k:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def plot_feature_importance(
        self, 
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """Визуализация важности признаков"""
        
        importance_df = self.get_feature_importance(top_k)
        
        if importance_df.empty:
            logger.warning("⚠️ Нет данных о важности признаков")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(
            range(len(importance_df)), 
            importance_df['importance'],
            color='skyblue',
            alpha=0.7
        )
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top {top_k} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Добавление значений
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График важности признаков сохранен: {save_path}")
        else:
            plt.show()
    
    def plot_correlation_heatmap(
        self,
        features: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Тепловая карта корреляций"""
        
        if self.metadata_.correlation_matrix is None:
            logger.warning("⚠️ Корреляционная матрица не вычислена")
            return
        
        corr_matrix = self.metadata_.correlation_matrix
        
        if features:
            available_features = [f for f in features if f in corr_matrix.columns]
            corr_matrix = corr_matrix.loc[available_features, available_features]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Корреляционная матрица сохранена: {save_path}")
        else:
            plt.show()
    
    def save_engineer(self, path: Union[str, Path]):
        """Сохранение feature engineer"""
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение конфигурации
        config_dict = {
            'enable_technical_indicators': self.config.enable_technical_indicators,
            'enable_statistical_features': self.config.enable_statistical_features,
            'enable_lag_features': self.config.enable_lag_features,
            'enable_interaction_features': self.config.enable_interaction_features,
            'enable_volatility_features': self.config.enable_volatility_features,
            'technical_periods': self.config.technical_periods,
            'statistical_windows': self.config.statistical_windows,
            'max_lags': self.config.max_lags,
            'scaling_method': self.config.scaling_method.value,
            # ... добавить другие параметры
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Сохранение метаданных
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.metadata_.to_dict(), f, indent=2, default=str)
        
        # Сохранение скейлеров
        if self.scalers_:
            joblib.dump(self.scalers_, path / "scalers.pkl")
        
        # Сохранение feature selector
        if self.feature_selector_:
            joblib.dump(self.feature_selector_, path / "feature_selector.pkl")
        
        logger.info(f"💾 Feature Engineer сохранен в {path}")
    
    @classmethod
    def load_engineer(cls, path: Union[str, Path]) -> 'FeatureEngineer':
        """Загрузка feature engineer"""
        
        path = Path(path)
        
        # Загрузка конфигурации
        with open(path / "config.json", 'r') as f:
            config_dict = json.load(f)
        
        # Создание конфигурации
        config = FeatureConfig(**config_dict)
        
        # Создание экземпляра
        engineer = cls(config)
        
        # Загрузка метаданных
        with open(path / "metadata.json", 'r') as f:
            metadata_dict = json.load(f)
        
        engineer.metadata_ = FeatureMetadata(**metadata_dict)
        
        # Загрузка скейлеров
        scalers_path = path / "scalers.pkl"
        if scalers_path.exists():
            engineer.scalers_ = joblib.load(scalers_path)
        
        # Загрузка feature selector
        selector_path = path / "feature_selector.pkl"
        if selector_path.exists():
            engineer.feature_selector_ = joblib.load(selector_path)
        
        engineer.is_fitted_ = True
        
        logger.info(f"📂 Feature Engineer загружен из {path}")
        
        return engineer


# Utility functions для быстрых вычислений
@jit(nopython=True, parallel=True)
def fast_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Быстрое вычисление скользящего среднего"""
    
    n = len(data)
    result = np.empty(n)
    
    for i in prange(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(data[i-window+1:i+1])
    
    return result


@jit(nopython=True)
def fast_percentage_change(data: np.ndarray) -> np.ndarray:
    """Быстрое вычисление процентного изменения"""
    
    n = len(data)
    result = np.empty(n)
    result[0] = 0.0
    
    for i in range(1, n):
        if data[i-1] != 0:
            result[i] = (data[i] - data[i-1]) / data[i-1]
        else:
            result[i] = 0.0
    
    return result


if __name__ == "__main__":
    # Пример использования
    logger.info("🧪 Тестирование Feature Engineer...")
    
    # Создание синтетических данных
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Синтетические данные о цене временных рядовы
    price_data = pd.DataFrame(index=dates)
    price_data['open'] = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    price_data['high'] = price_data['open'] + np.random.exponential(0.5, n_samples)
    price_data['low'] = price_data['open'] - np.random.exponential(0.5, n_samples)
    price_data['close'] = price_data['open'] + np.random.randn(n_samples) * 0.5
    price_data['volume'] = np.random.exponential(1000, n_samples)
    
    # Целевая переменная - будущее изменение цены
    target = price_data['close'].pct_change(periods=5).shift(-5)  # 5-period forward return
    
    # Создание feature engineer
    config = FeatureConfig(
        enable_technical_indicators=True,
        enable_statistical_features=True,
        enable_lag_features=True,
        enable_interaction_features=False,  # Отключаем для быстроты
        enable_volatility_features=True,
        feature_selection_k=50
    )
    
    engineer = FeatureEngineer(config)
    
    # Обучение и трансформация
    X_transformed = engineer.fit_transform(price_data, target.dropna())
    
    logger.info(f"📊 Исходных признаков: {len(price_data.columns)}")
    logger.info(f"📊 Создано признаков: {len(X_transformed.columns)}")
    
    # Важность признаков
    importance = engineer.get_feature_importance(top_k=10)
    if not importance.empty:
        logger.info(f"🎯 Топ-3 признака: {importance['feature'].iloc[:3].tolist()}")
    
    logger.info("✅ Тестирование Feature Engineer завершено!")