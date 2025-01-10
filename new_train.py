import os
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
import logging
from collections import Counter
import datetime

# Определяем пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'ST14000NM001G.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
LOGS_PATH = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(LOGS_PATH, 'training.log'))
file_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def load_data() -> pl.DataFrame:
    data = pl.read_csv(DATA_PATH)
    logging.info(f"Колонки в исходных данных: {data.columns}")
    
    # Парсим столбец "date" как дату с форматом YYYY-MM-DD (в старых версиях Polars нет именованных аргументов)
    data = data.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    
    return data

def label_future_failure_in_30_days(df: pl.DataFrame) -> pl.DataFrame:
    """
    Сортирует df по date, и для каждой строки проверяет, будет ли в течение 30 дней запись с failure=1.
    Добавляет столбец failure_next_30d (0/1).
    """
    df = df.sort("date")
    date_col = df["date"]
    fail_col = df["failure"]

    future_fail = []
    n = len(df)
    for i in range(n):
        d_i = date_col[i]  # уже должен быть тип Date
        cutoff = d_i + datetime.timedelta(days=30)
        
        found_fail = 0
        for j in range(i+1, n):
            if date_col[j] <= cutoff:
                if fail_col[j] == 1:
                    found_fail = 1
                    break
            else:
                break
        future_fail.append(found_fail)

    # Вместо with_column используем hstack:
    return df.hstack(
        pl.DataFrame({"failure_next_30d": future_fail})
    )

def group_by_and_apply(data: pl.DataFrame, group_col: str, func) -> pl.DataFrame:
    """
    Аналог group_by(...).apply(...)
    """
    unique_vals = data.select(group_col).unique().to_series().to_list()
    results = []
    for val in unique_vals:
        subset = data.filter(pl.col(group_col) == val)
        subset_result = func(subset)
        results.append(subset_result)
    return pl.concat(results)

def prepare_dataset(data: pl.DataFrame):
    try:
        # Вместо data.group_by("serial_number").apply(...), используем самодельную функцию:
        data = group_by_and_apply(
            data,
            group_col="serial_number",
            func=label_future_failure_in_30_days
        )
        
        # Теперь data имеет столбец 'failure_next_30d'
        serial_numbers = data.get_column("serial_number")
        y = data.get_column("failure_next_30d")

        # Удалим из X то, что не нужно для обучения
        columns_to_drop = ["serial_number", "failure", "failure_next_30d", "model"]
        # columns_to_drop.append("date") если не хотим использовать дату
        columns_to_drop = [col for col in columns_to_drop if col in data.columns]

        X = data.drop(columns_to_drop)

        # Преобразуем capacity_bytes в целое
        if "capacity_bytes" in X.columns:
            X = X.with_columns(pl.col("capacity_bytes").cast(pl.Int64))

        return X, y, serial_numbers
    except Exception as e:
        logging.error("Ошибка при формировании датасета (failure_next_30d):", exc_info=True)
        raise

def log_transform(X: pl.DataFrame) -> pl.DataFrame:
    """
    Логарифмирование столбцов со SMART-параметрами (если есть).
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
    """Обучение LightGBM."""
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
    """Обучение CatBoost."""
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
    - Группы = serial_number
    - Стратификация по y (failure_next_30d)
    - SMOTE только для train
    """
    try:
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        groups_np = groups.to_numpy() if not isinstance(groups, np.ndarray) else groups

        lightgbm_models = []
        catboost_models = []

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_np, y_np, groups=groups_np), start=1):
            logging.info(f"\nФолд {fold_idx}/{n_splits}")

            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]

            class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
            logging.info(f"Распределение классов (до SMOTE) в train фолда {fold_idx}: {class_dist}")

            smote = SMOTE(random_state=42, n_jobs=-1)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            class_dist_after = dict(zip(*np.unique(y_train_res, return_counts=True)))
            logging.info(f"Распределение классов (после SMOTE) в train фолда {fold_idx}: {class_dist_after}")

            # LightGBM
            lgb_model = train_lightgbm(X_train_res, y_train_res, X_test, y_test)
            lightgbm_models.append(lgb_model)

            # CatBoost
            cat_model = train_catboost(X_train_res, y_train_res, X_test, y_test)
            catboost_models.append(cat_model)

        return lightgbm_models, catboost_models
    except Exception as e:
        logging.error("Ошибка при обучении моделей с кросс-валидацией:", exc_info=True)
        raise

def main():
    try:
        logging.info("Загрузка сырых данных")
        df = load_data()

        logging.info("Создаём метку 'умрёт в течение 30 дней'")
        X, y, serial_numbers = prepare_dataset(df)

        X = log_transform(X)

        logging.info("\nРаспределение классов (failure_next_30d):")
        class_counts = y.value_counts(sort=True)
        logging.info(f"{class_counts}")

        logging.info("\nСтатистика X.describe():")
        feature_stats = X.describe()
        logging.info(f"{feature_stats}")

        logging.info("\nНачинаем обучение (StratifiedGroupKFold)")
        lightgbm_models, catboost_models = train_models_iteratively(X, y, serial_numbers, n_splits=5)

        for i, (lgb_model, cat_model) in enumerate(zip(lightgbm_models, catboost_models), start=1):
            lgb_path = os.path.join(MODEL_PATH, f'lightgbm_model_{i}.pkl')
            cat_path = os.path.join(MODEL_PATH, f'catboost_model_{i}.pkl')
            joblib.dump(lgb_model, lgb_path)
            joblib.dump(cat_model, cat_path)
            logging.info(f"Сохранена LightGBM модель {i} в {lgb_path}")
            logging.info(f"Сохранена CatBoost модель {i} в {cat_path}")

        logging.info("Обучение завершено успешно.")
    except Exception as e:
        logging.error("Ошибка в главной функции:", exc_info=True)
        print("Произошла ошибка. См. лог-файл для деталей.")

if __name__ == "__main__":
    main()
