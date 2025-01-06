import os
import polars as pl
import joblib
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Определяем пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'ST14000NM001G.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
LOGS_PATH = os.path.join(BASE_DIR, 'logs')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics')

# Создаем директории
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(METRICS_PATH, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_PATH, 'evaluation.log')),
        logging.StreamHandler()
    ]
)

def load_data():
    """Загрузка данных."""
    try:
        data = pl.read_csv(DATA_PATH)
        
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

def preprocess_data(X):
    """Предобработка данных."""
    try:
        # Логарифмируем SMART-атрибуты
        smart_cols = [col for col in X.columns if 'smart' in col.lower()]
        if smart_cols:
            X = X.with_columns([
                pl.col(col).log1p().alias(col) for col in smart_cols
            ])
        return X
    except Exception as e:
        logging.error("Ошибка при предобработке данных:", exc_info=True)
        raise

def load_models():
    """Загрузка сохранённых моделей."""
    try:
        lightgbm_models = []
        catboost_models = []
        
        for i in range(1, 6):  # Предполагаем 5 моделей каждого типа
            lgb_path = os.path.join(MODEL_PATH, f'lightgbm_model_{i}.pkl')
            cat_path = os.path.join(MODEL_PATH, f'catboost_model_{i}.pkl')
            
            if os.path.exists(lgb_path):
                lightgbm_models.append(joblib.load(lgb_path))
            if os.path.exists(cat_path):
                catboost_models.append(joblib.load(cat_path))
        
        return lightgbm_models, catboost_models
    except Exception as e:
        logging.error("Ошибка при загрузке моделей:", exc_info=True)
        raise

def plot_curves(y_true, y_pred_proba):
    """Построение ROC и PR кривых."""
    try:
        # ROC кривая
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 5))
        
        # ROC кривая
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # PR кривая
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_PATH, 'pr_curve.png'))
        plt.close()
        
    except Exception as e:
        logging.error("Ошибка при построении графиков:", exc_info=True)
        raise

def evaluate_models(X, y, lightgbm_models, catboost_models):
    """Оценка моделей."""
    try:
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        
        # Получение предсказаний
        lgb_preds = np.mean([model.predict_proba(X_np)[:, 1] for model in lightgbm_models], axis=0)
        cat_preds = np.mean([model.predict_proba(X_np)[:, 1] for model in catboost_models], axis=0)
        
        # Усреднение предсказаний
        y_pred_proba = (lgb_preds + cat_preds) / 2
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Вычисление метрик
        metrics = {
            'AUC': roc_auc_score(y_np, y_pred_proba),
            'Precision': precision_score(y_np, y_pred),
            'Recall': recall_score(y_np, y_pred),
            'F1': f1_score(y_np, y_pred),
            'Confusion Matrix': confusion_matrix(y_np, y_pred).tolist()
        }
        
        # Сохранение метрик в файл
        with open(os.path.join(METRICS_PATH, 'metrics.txt'), 'w') as f:
            for metric, value in metrics.items():
                if metric != 'Confusion Matrix':
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}:\n{np.array2string(np.array(value))}\n")
        
        # Построение графиков
        plot_curves(y_np, y_pred_proba)
        
        # Вывод метрик в консоль
        logging.info("Метрики на всём наборе данных:")
        for metric, value in metrics.items():
            if metric != 'Confusion Matrix':
                logging.info(f"{metric}: {value:.4f}")
            else:
                logging.info(f"{metric}:\n{np.array(value)}")
                
        return metrics
        
    except Exception as e:
        logging.error("Ошибка при оценке моделей:", exc_info=True)
        raise

def main():
    """Главная функция."""
    try:
        logging.info("Начало оценки моделей")
        
        # Загрузка данных
        X, y = load_data()
        
        # Предобработка данных
        X = preprocess_data(X)
        
        # Загрузка моделей
        lightgbm_models, catboost_models = load_models()
        
        if not lightgbm_models or not catboost_models:
            raise ValueError("Не найдены сохранённые модели")
        
        # Оценка моделей
        evaluate_models(X, y, lightgbm_models, catboost_models)
        
        logging.info("Оценка моделей завершена успешно")
        
    except Exception as e:
        logging.error("Ошибка в главной функции:", exc_info=True)
        raise

if __name__ == "__main__":
    main()