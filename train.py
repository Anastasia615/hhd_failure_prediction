import os
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import logging
from collections import Counter

# Определяем пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'ST14000NM001G.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
LOGS_PATH = os.path.join(BASE_DIR, 'logs')

# Создаем директории если их нет
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Настройка логирования с выводом в файл и консоль
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Обработчик для файла
file_handler = logging.FileHandler(os.path.join(LOGS_PATH, 'training.log'))
file_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Обработчик для консоли
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def load_data():
    """Загрузка и подготовка данных с использованием Polars."""
    try:
        data = pl.read_csv(DATA_PATH)
        logging.info(f"Столбцы в данных: {data.columns}")
        
        # Удаляем строковые столбцы
        columns_to_drop = ['date', 'serial_number', 'model']
        X = data.drop(columns_to_drop + ['failure'])
        
        # Преобразуем в числовой формат
        X = X.with_columns(pl.col('capacity_bytes').cast(pl.Int64))
        
        y = data.get_column('failure')
        return X, y
    except Exception as e:
        logging.error("Ошибка при загрузке данных:", exc_info=True)
        raise

def preprocess_data(X, y, balance_once=True):
    """Предобработка данных с использованием Polars."""
    try:
        # Логарифмируем SMART-атрибуты
        smart_cols = [col for col in X.columns if 'smart' in col.lower()]
        if smart_cols:
            X = X.with_columns([
                pl.col(col).log1p().alias(col) for col in smart_cols
            ])
            logging.info(f"Логарифмированные столбцы: {smart_cols}")
        
        if balance_once:
            # Преобразуем в numpy для балансировки
            X_np = X.to_numpy()
            y_np = y.to_numpy()
            
            logging.info("Начало балансировки данных с использованием SMOTE")
            # Используем метод SMOTE с параллельной обработкой
            sampler = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)
            X_resampled, y_resampled = sampler.fit_resample(X_np, y_np)
            logging.info(f"После SMOTE: {Counter(y_resampled)}")
            
            # Возвращаем обратно в Polars
            try:
                X = pl.DataFrame(X_resampled, schema=X.schema)
                y = pl.Series(y_resampled)
                logging.info("Преобразование обратно в Polars DataFrame успешно")
            except Exception as e:
                logging.error("Ошибка при преобразовании обратно в Polars DataFrame:", exc_info=True)
                raise
        
        return X, y
    except Exception as e:
        logging.error("Ошибка при предобработке данных:", exc_info=True)
        raise

def train_models_iteratively(X, y, n_splits=5):
    """Обучение моделей с использованием кросс-валидации."""
    try:
        X_balanced, y_balanced = preprocess_data(X, y, balance_once=True)
        
        # Конвертируем в numpy для sklearn
        X_np = X_balanced.to_numpy()
        y_np = y_balanced.to_numpy()
        
        lightgbm_models = []
        catboost_models = []
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for i, (train_idx, test_idx) in enumerate(skf.split(X_np, y_np), 1):
            logging.info(f"\nИтерация {i}/{n_splits}")
            
            X_train = X_np[train_idx]
            X_test = X_np[test_idx]
            y_train = y_np[train_idx]
            y_test = y_np[test_idx]
            
            class_distribution = dict(zip(*np.unique(y_train, return_counts=True)))
            class_distribution_normalized = {k: v / len(y_train) for k, v in class_distribution.items()}
            logging.info(f"Распределение классов в фолде: {class_distribution_normalized}")
            
            lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
            cat_model = train_catboost(X_train, y_train, X_test, y_test)
            
            lightgbm_models.append(lgb_model)
            catboost_models.append(cat_model)
        
        return lightgbm_models, catboost_models
    except Exception as e:
        logging.error("Ошибка при обучении моделей с кросс-валидацией:", exc_info=True)
        raise

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Обучение LightGBM с улучшенными параметрами"""
    try:
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',  # важно для несбалансированных данных
            random_state=42,
            verbose=-1
        )
        
        # Используем callbacks для early stopping без передачи logger
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)  # Убрали параметр `logger`
            ]
        )
        
        logging.info("LightGBM модель успешно обучена")
        return model
    except Exception as e:
        logging.error("Ошибка при обучении LightGBM модели:", exc_info=True)
        raise

def train_catboost(X_train, y_train, X_test, y_test):
    """Обучение модели CatBoost."""
    try:
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=5,              # Уменьшаем глубину
            l2_leaf_reg=5,       # Увеличиваем регуляризацию
            min_data_in_leaf=20, # Минимум примеров в листе
            random_seed=42,
            verbose=False
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False
        )
        logging.info("CatBoost модель успешно обучена")
        return model
    except Exception as e:
        logging.error("Ошибка при обучении CatBoost модели:", exc_info=True)
        raise

def main():
    """Главная функция для обучения моделей."""
    try:
        logging.info("Начало загрузки данных")
        X, y = load_data()
        X, y = preprocess_data(X, y)
        
        logging.info("\nРаспределение классов:")
        class_counts = y.value_counts(normalize=True)
        logging.info(f"{class_counts}")
        
        logging.info("\nСтатистика признаков:")
        feature_stats = X.describe()
        logging.info(f"{feature_stats}")
        
        logging.info("Начало обучения моделей с кросс-валидацией")
        lightgbm_models, catboost_models = train_models_iteratively(X, y, n_splits=5)
        
        # Сохраняем все модели
        for i, (lgb_model, cat_model) in enumerate(zip(lightgbm_models, catboost_models), 1):
            lgb_path = os.path.join(MODEL_PATH, f'lightgbm_model_{i}.pkl')
            cat_path = os.path.join(MODEL_PATH, f'catboost_model_{i}.pkl')
            joblib.dump(lgb_model, lgb_path)
            joblib.dump(cat_model, cat_path)
            logging.info(f"Сохранена LightGBM модель {i} по пути {lgb_path}")
            logging.info(f"Сохранена CatBoost модель {i} по пути {cat_path}")
        
        logging.info("Обучение моделей завершено успешно")
    except Exception as e:
        logging.error("Ошибка в главной функции:", exc_info=True)
        print("Произошла ошибка во время выполнения. Проверьте лог файл для деталей.")

if __name__ == "__main__":
    main()