# Uplift Modeling + Contact Policy + Propensity & Clustering

Проект решает три связанные задачи маркетингового ML на классическом датасете **Hillstrom (Customer Retention)**:

1. **Uplift-моделирование** — оценка индивидуального прироста конверсии (CATE) от маркетингового предложения (X-Learner).
2. **Контактная политика** — правила коммуникации («Активно / Стандарт / Пауза») на основе uplift-score + экономическая оценка таргетинга.
3. **Propensity to buy + кластеризация** — склонность к покупке и сегментация клиентов.

---

## Бизнес-ценность

- Сокращение маркетингового бюджета за счёт отправки предложений только «убеждаемым» клиентам (persuadables).
- Повышение ROI кампаний.
- Улучшение клиентского опыта (меньше бесполезных контактов).

---

## Структура проекта

```
Uplift-Project/
├── data/
│   └── customer_retention.csv          # датасет (уже лежит в репозитории)
├── notebooks/
│   ├── 01_eda_and_preparation.ipynb    # EDA + feature engineering + split
│   ├── 02_uplift_modeling.ipynb        # X-Learner, AUUC, Qini
│   ├── 03_contact_policy_and_roi.ipynb # политика + экономика сегментов
│   └── 04_propensity_and_clustering.ipynb
├── src/
│   ├── preprocessing.py
│   ├── uplift.py                       # класс XLearner
│   ├── policy.py                       # контактная политика + economics
│   └── utils.py                        # AUUC, Qini, helpers
├── models/                             # сохранённые модели (joblib)
├── outputs/                            # предсказания, таблицы, метрики
├── figures/                            # графики
├── requirements.txt
└── README.md
```

---

## Быстрый старт

```bash
# 1. Клонировать / распаковать
cd Uplift-Project

# 2. Создать окружение
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить ноутбуки по порядку (01 → 02 → 03 → 04)
jupyter notebook notebooks/
```

Данные уже находятся в `data/customer_retention.csv`.  
Если нужно скачать оригинал: [Kaggle — Customer Retention](https://www.kaggle.com/datasets/davinwijaya/customer-retention).

---

## Методология (кратко)

### Uplift (X-Learner)
1. Propensity model \(e(x) = P(T=1 \mid X)\)
2. Outcome models \(\mu_1(x)\), \(\mu_0(x)\)
3. Pseudo-outcomes (imputed treatment effects)
4. Effect models \(\tau_1(x)\), \(\tau_0(x)\)
5. Final uplift = \(e(x)\tau_1(x) + (1-e(x))\tau_0(x)\)

Метрики: **AUUC**, **Qini-кривая**.

### Контактная политика
Строится **на uplift-score**, а не на сырой вероятности покупки:

| Политика   | Условие                  | Смысл                          |
|------------|--------------------------|--------------------------------|
| Активно    | uplift ≥ high_threshold  | Persuadables                   |
| Стандарт   | между порогами           | Нейтральные                    |
| Пауза      | uplift ≤ low_threshold   | Sleeping dogs / sure things    |

Экономика сегментов считается по формуле:

```
incremental_conversions = n_selected × observed_uplift
profit = incremental_conversions × MARGIN − n_selected × COST
```

### Кластеризация
K-Means обучается **только на train**, предсказания — на test (нет утечки).

---

## Основные артефакты после запуска

| Файл | Описание |
|------|----------|
| `models/xlearner.joblib` | Обученный X-Learner |
| `models/preprocessor.joblib` | ColumnTransformer |
| `models/propensity_to_buy.joblib` | Logistic Regression |
| `models/kmeans.joblib` | Кластеризация |
| `outputs/segment_economics.csv` | Таблица uplift + profit по top-% |
| `outputs/contact_policy_test.csv` | Политика на тестовой выборке |
| `figures/*.png` | Все ключевые графики |

---

---

## Ограничения и дальнейшие улучшения

- Датасет синтетический/учебный (Hillstrom). На реальных данных нужна проверка assumption of unconfoundedness.
- Пороги политики и константы MARGIN/COST нужно калибровать под конкретный бизнес.
- Можно добавить сравнение meta-learners (S/T/X), Causal Forest, или перейти на EconML / CausalML.
- Для production — обернуть в FastAPI + простой дашборд (см. также репозиторий MLOps-uplift-marketing того же автора).

---

**Автор**: Mamaeva Bariyat  
**Стек**: Python, pandas, scikit-learn, LightGBM, Optuna-ready, joblib
