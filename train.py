import os
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
# Если нет StratifiedGroupKFold в вашей версии sklearn, обновитесь до 1.3+
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import numpy as np
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
        # Читаем csv
        data = pl.read_csv(DATA_PATH)
        logging.info(f"Столбцы в данных: {data.columns}")
        
        # Сохраним serial_number как отдельную переменную, чтобы потом использовать в группах
        serial_numbers = data.get_column('serial_number')
        
        # Целевая переменная
        y = data.get_column('failure')
        
        # Удаляем ненужные столбцы (но НЕ удаляем 'serial_number' до того, как сохраним его)
        columns_to_drop = ['date', 'serial_number', 'model', 'failure']
        X = data.drop(columns_to_drop)
        
        # Преобразуем capacity_bytes в целое (если нужно)
        if 'capacity_bytes' in X.columns:
            X = X.with_columns(pl.col('capacity_bytes').cast(pl.Int64))
        
        return X, y, serial_numbers
    except Exception as e:
        logging.error("Ошибка при загрузке данных:", exc_info=True)
        raise

def log_transform(X):
    """
    Логарифмирование столбцов со SMART-параметрами.
    Отдельно выделено, чтобы быть вызванным один раз.
    """
    try:
        smart_cols = [col for col in X.columns if 'smart' in col.lower()]
        if smart_cols:
            X = X.with_columns([
                pl.col(col).log1p().alias(col) for col in smart_cols
            ])
            logging.info(f"Логарифмированные столбцы: {smart_cols}")
        return X
    except Exception as e:
        logging.error("Ошибка при логарифмировании SMART-атрибутов:", exc_info=True)
        raise

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Обучение LightGBM с «оптимизированными» параметрами."""
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
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
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
            depth=5,
            l2_leaf_reg=5,
            min_data_in_leaf=20,
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

def train_models_iteratively(X, y, groups, n_splits=5):
    """
    Обучение моделей с использованием StratifiedGroupKFold:
    - Группы = serial_number, чтобы один и тот же диск не попадал в train и test одновременно.
    - Стратификация по целевой переменной y.
    - Балансировка SMOTE выполняется ТОЛЬКО для train, чтобы избежать утечки данных.
    """
    try:
        # Преобразуем X, y, groups в numpy для split
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        groups_np = groups.to_numpy() if not isinstance(groups, np.ndarray) else groups

        lightgbm_models = []
        catboost_models = []
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_np, y_np, groups=groups_np), start=1):
            logging.info(f"\nФолд {fold_idx}/{n_splits}")
            
            # Делим на train и test
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]
            
            # Посмотрим на баланс классов в этом фолде (только среди train)
            class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
            logging.info(f"Распределение классов (до SMOTE) в train фолда {fold_idx}: {class_dist}")
            
            # Балансировка SMOTE ТОЛЬКО для обучающей выборки
            # Можно заменить на BorderlineSMOTE / SMOTETomek / SVMSMOTE и т.д.
            smote = SMOTE(random_state=42, n_jobs=-1)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            class_dist_after = dict(zip(*np.unique(y_train_res, return_counts=True)))
            logging.info(f"Распределение классов (после SMOTE) в train фолда {fold_idx}: {class_dist_after}")
            
            # Обучение LightGBM
            lgb_model = train_lightgbm(X_train_res, y_train_res, X_test, y_test)
            lightgbm_models.append(lgb_model)
            
            # Обучение CatBoost
            cat_model = train_catboost(X_train_res, y_train_res, X_test, y_test)
            catboost_models.append(cat_model)
        
        return lightgbm_models, catboost_models
    except Exception as e:
        logging.error("Ошибка при обучении моделей с кросс-валидацией:", exc_info=True)
        raise

def main():
    """Главная функция для обучения моделей."""
    try:
        logging.info("Начало загрузки данных")
        X, y, serial_numbers = load_data()
        
        # Логарифмируем SMART-параметры
        X = log_transform(X)
        
        # Выведем статистику по метке (failure)
        logging.info("\nРаспределение классов (по всему датасету):")
        # Используем sort=True, чтобы сразу получить отсортированные значения
        class_counts = y.value_counts(sort=True)
        logging.info(f"{class_counts}")
        
        logging.info("\nСтатистика признаков (X.describe()):")
        feature_stats = X.describe()
        logging.info(f"{feature_stats}")
        
        # Обучаем модели с кросс-валидацией по группам (serial_number)
        logging.info("\nНачало обучения моделей с StratifiedGroupKFold...")
        lightgbm_models, catboost_models = train_models_iteratively(
            X, y, serial_numbers, n_splits=5
        )
        
        # Сохраняем все модели
        for i, (lgb_model, cat_model) in enumerate(zip(lightgbm_models, catboost_models), start=1):
            lgb_path = os.path.join(MODEL_PATH, f'lightgbm_model_{i}.pkl')
            cat_path = os.path.join(MODEL_PATH, f'catboost_model_{i}.pkl')
            joblib.dump(lgb_model, lgb_path)
            joblib.dump(cat_model, cat_path)
            logging.info(f"Сохранена LightGBM модель {i} по пути {lgb_path}")
            logging.info(f"Сохранена CatBoost модель {i} по пути {cat_path}")
        
        logging.info("Обучение моделей завершено успешно")
    except Exception as e:
        logging.error("Ошибка в главной функции:", exc_info=True)
        print("Произошла ошибка во время выполнения. Проверьте лог-файл для деталей.")

if __name__ == "__main__":
    main()
