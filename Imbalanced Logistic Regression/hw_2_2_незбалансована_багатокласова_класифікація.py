

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import math
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from sklearn import metrics
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

def split_train_test(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
  """
    Splits a dataset into training and test sets
    using stratified sampling based on the target variable.

    Args:
        df (pd.DataFrame): Full dataset containing features and target.
        target (str): Target column name used for stratification.

    Returns:
        X_train (pd.DataFrame): Training subset.
        X_test (pd.DataFrame): Test subset.
    """
  X_train, X_test = train_test_split(
        df,
        test_size=test_size,  # 80% train, 20% validation
        stratify=df[target],
        random_state=random_state
    )

  return X_train, X_test

def extract_target(df: pd.DataFrame, target: str):
    """
    Separates input features and target variable from a dataset.

    Args:
    df : pd.DataFrame
        The full raw containing both feature columns and the target column.
    target : str
        Name of the target column to be extracted from the dataset.

    Return:
    input_features : A copy of the dataset without the target column (feature matrix X).
    target : A copy of the target column (label vector y).
    """

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    return X, y

def drop_columns(X: pd.DataFrame, columns_to_drop):
  """
    Drop specified columns from train, validation, or test splits.
    """
  return X.drop(columns=columns_to_drop, errors="ignore")

def encode_target_train(y_train: pd.Series):
  """
  Use LabelEncoder() to encode target variable
  """
  target_encoder = LabelEncoder()
  y_train_encoded = target_encoder.fit_transform(y_train)

  return y_train_encoded, target_encoder

def target_transofrm_test(y_test: pd.Series, target_encoder: LabelEncoder):
  """
  Target transform for test set.
  """
  return target_encoder.transform(y_test)

def detect_missing_values_cols(df: pd.DataFrame, threshold: float = 0.6):
    """
    Returns:
    - columns containing missing values
    - columns where missing percentage exceeds threshold
    """
    cols_with_na = []
    cols_to_delete = []

    for column in df.columns:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            cols_with_na.append(column)

            missing_percentage = missing_count / len(df)

            if missing_percentage > threshold:
                cols_to_delete.append(column)

    return cols_with_na, cols_to_delete

def impute_missing_values(X_train: pd.DataFrame):

  X_train_copy = X_train.copy()
  numeric_features = X_train_copy.select_dtypes(include=["number"]).columns.tolist()
  categorical_features = X_train_copy.select_dtypes(include=["object"]).columns.tolist()

  numeric_imputer = SimpleImputer(strategy="median")
  categorical_imputer = SimpleImputer(strategy="most_frequent")

  if numeric_features:
    X_train_copy[numeric_features] = numeric_imputer.fit_transform(X_train_copy[numeric_features])
  if categorical_features:
    X_train_copy[categorical_features] = categorical_imputer.fit_transform(X_train_copy[categorical_features])

  return X_train_copy, numeric_features, categorical_features, numeric_imputer, categorical_imputer

def impute_missing_values_transform(X_test: pd.DataFrame, numeric_features: list, categorical_features: list, numeric_imputer: SimpleImputer, categorical_imputer: SimpleImputer):

  X_test_copy = X_test.copy()

  for column in numeric_features:
    if column not in X_test_copy.columns:
            X_test_copy[column] = np.nan

  for column in categorical_features:
    if column not in X_test_copy.columns:
            X_test_copy[column] = np.nan

  if numeric_features:
        X_test_copy[numeric_features] = numeric_imputer.transform(X_test_copy[numeric_features])

  if categorical_features:
        X_test_copy[categorical_features] = categorical_imputer.transform(X_test_copy[categorical_features])

  return X_test_copy

def encode_categories_train(X_train: pd.DataFrame, numeric_features: list, categorical_features: list, encoder_type: str = "OneHotEncoder"):
    """
    Supports different types of encoding for categorical variables:
    encoder_type = "OneHotEncoder" or "OrdinalEncoder"
    """
    X_train_copy = X_train.copy()

    if not categorical_features:
        X_train_encoded = X_train_copy[numeric_features].copy()
        feature_columns = X_train_encoded.columns.tolist()
        return X_train_encoded, None, feature_columns

    if encoder_type == "OneHotEncoder":

        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        encoder.fit(X_train_copy[categorical_features])

        encoded = encoder.transform(X_train_copy[categorical_features])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_features),
            index=X_train_copy.index
        )

        X_train_encoded = pd.concat(
            [X_train_copy[numeric_features], encoded_df],
            axis=1
        )

    elif encoder_type == "OrdinalEncoder":

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        encoder.fit(X_train_copy[categorical_features])

        encoded = encoder.transform(X_train_copy[categorical_features])

        encoded_df = pd.DataFrame(
            encoded,
            columns=categorical_features,
            index=X_train_copy.index
        )

        X_train_encoded = pd.concat(
            [X_train_copy[numeric_features], encoded_df],
            axis=1
        )

    else:
        raise ValueError('encoder_type must be "OneHotEncoder" or "OrdinalEncoder"')

    feature_columns = X_train_encoded.columns.tolist()

    return X_train_encoded, encoder, feature_columns

def encode_categories_test(X_test: pd.DataFrame, numeric_features: list, categorical_features: list, encoder, feature_columns: list):

    X_test_copy = X_test.copy()

    if categorical_features:

        encoded = encoder.transform(X_test_copy[categorical_features])

        # OneHotEncoder option
        if isinstance(encoder, OneHotEncoder):

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(categorical_features),
                index=X_test_copy.index
            )
        # Ordinal Encoder option

        elif isinstance(encoder, OrdinalEncoder):

            encoded_df = pd.DataFrame(
                encoded,
                columns=categorical_features,
                index=X_test_copy.index
            )

        else:
            raise ValueError("Unsupported encoder type")

        X_test_encoded = pd.concat(
            [X_test_copy[numeric_features], encoded_df],
            axis=1
        )

    else:
        X_test_encoded = X_test_copy[numeric_features].copy()

    for col in feature_columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0.0

    X_test_encoded = X_test_encoded[feature_columns]

    return X_test_encoded

def scale_numeric_train(X_train_encoded: pd.DataFrame):
  scaler = MinMaxScaler()
  X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)

  return X_train_scaled, scaler

def scale_numeric_test(X_test_encoded: pd.DataFrame):
  X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)

  return X_test_scaled

"""### Опис задачі і даних

**Контекст**

В цьому ДЗ ми працюємо з даними про сегментацію клієнтів.

Сегментація клієнтів – це практика поділу бази клієнтів на групи індивідів, які схожі між собою за певними критеріями, що мають значення для маркетингу, такими як вік, стать, інтереси та звички у витратах.

Компанії, які використовують сегментацію клієнтів, виходять з того, що кожен клієнт є унікальним і що їхні маркетингові зусилля будуть більш ефективними, якщо вони орієнтуватимуться на конкретні, менші групи зі зверненнями, які ці споживачі вважатимуть доречними та які спонукатимуть їх до купівлі. Компанії також сподіваються отримати глибше розуміння уподобань та потреб своїх клієнтів з метою виявлення того, що кожен сегмент цінує найбільше, щоб точніше адаптувати маркетингові матеріали до цього сегменту.

**Зміст**.

Автомобільна компанія планує вийти на нові ринки зі своїми існуючими продуктами (P1, P2, P3, P4 і P5). Після інтенсивного маркетингового дослідження вони дійшли висновку, що поведінка нового ринку схожа на їхній існуючий ринок.

На своєму існуючому ринку команда з продажу класифікувала всіх клієнтів на 4 сегменти (A, B, C, D). Потім вони здійснювали сегментовані звернення та комунікацію з різними сегментами клієнтів. Ця стратегія працювала для них надзвичайно добре. Вони планують використати ту саму стратегію на нових ринках і визначили 2627 нових потенційних клієнтів.

Ви маєте допомогти менеджеру передбачити правильну групу для нових клієнтів.

В цьому ДЗ використовуємо дані `customer_segmentation_train.csv`[скачати дані](https://drive.google.com/file/d/1VU1y2EwaHkVfr5RZ1U4MPWjeflAusK3w/view?usp=sharing). Це `train.csv`з цього [змагання](https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation/data?select=train.csv)

# Main

**Завдання 1.** Завантажте та підготуйте датасет до аналізу. Виконайте обробку пропущених значень та необхідне кодування категоріальних ознак. Розбийте на тренувальну і тестувальну вибірку, де в тесті 20%. Памʼятаємо, що весь препроцесинг ліпше все ж тренувати на тренувальній вибірці і на тестувальній лише використовувати вже натреновані трансформери.
Але в даному випадку оскільки значень в категоріях небагато, можна зробити обробку і на оригінальних даних, а потім розбити - це простіше. Можна також реалізувати процесинг і тренування моделі з пайплайнами. Обирайте як вам зручніше.
"""

train_raw_df = pd.read_csv("drive/MyDrive/MLcourse/customer_segmentation_train.csv")

train_raw_df.head()

X_train, X_test = split_train_test(train_raw_df, 'Segmentation')

X_train

X_train['Segmentation'].value_counts()

X_train, y_train = extract_target(X_train, "Segmentation")

X_train

y_train, target_encoder = encode_target_train(y_train)

y_train

cols_with_na, cols_to_delete = detect_missing_values_cols(X_train)

cols_with_na

X_train = drop_columns(X_train, "ID")

X_train

X_train.info()

X_train.isna().sum()

X_train, numeric_features, categorical_features, numeric_imputer, categorical_imputer = impute_missing_values(X_train)

X_train.info()

X_train.isna().sum()

X_train

exclusively_numeric = X_train.select_dtypes(include=["number"]).columns.tolist()

exclusively_numeric

# iterate through the list of all columns and if column in categorical, append in another list the index of this categorical varible
"""
columns_category = X_train.select_dtypes(include=["object"]).columns.tolist()
all_columns_list = X_train.columns.to_list()
columns_category_indices = []
for index in range(0, len(all_columns_list)):
  if all_columns_list[index] in columns_category:
    columns_category_indices.append(index)
"""

X_train, encoder, feature_columns = encode_categories_train(X_train, numeric_features, categorical_features, encoder_type='OneHotEncoder')

X_train

# scale
X_train, scaler = scale_numeric_train(X_train)

X_train

columns_list = X_train.columns.to_list()

columns_category_indices = []

# get all columns except 'exclusively_numeric'
for index in range(0, len(columns_list)):
  if columns_list[index] not in exclusively_numeric:
    columns_category_indices.append(index)

columns_category_indices # used later for SMOTENC resampling to pass

"""
LOAD THE TEST SET and preprocess
"""

X_test

X_test, y_test = extract_target(X_test, "Segmentation")

X_test

y_test = target_transofrm_test(y_test, target_encoder)
y_test

X_test = drop_columns(X_test, "ID")
X_test

X_test = impute_missing_values_transform(X_test, numeric_features, categorical_features, numeric_imputer, categorical_imputer)

X_test

X_test = encode_categories_test(X_test, numeric_features, categorical_features, encoder, feature_columns)

X_test

X_test = scale_numeric_test(X_test)

X_test

"""**Завдання 2. Важливо уважно прочитати все формулювання цього завдання до кінця!**

Застосуйте методи ресемплингу даних SMOTE та SMOTE-Tomek з бібліотеки imbalanced-learn до тренувальної вибірки. В результаті у Вас має вийти 2 тренувальних набори: з апсемплингом зі SMOTE, та з ресамплингом з SMOTE-Tomek.

Увага! В нашому наборі даних є як категоріальні дані, так і звичайні числові. Базовий SMOTE не буде правильно працювати з категоріальними даними, але є його модифікація, яка буде. Тому в цього завдання є 2 виконання

  1. Застосувати SMOTE базовий лише на НЕкатегоріальних ознаках.

  2. Переглянути інформацію про метод [SMOTENC](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTENC.html#imblearn.over_sampling.SMOTENC) і використати цей метод в цій задачі. За цей спосіб буде +3 бали за це завдання і він рекомендований для виконання.

  **Підказка**: аби скористатись SMOTENC треба створити змінну, яка містить індекси ознак, які є категоріальними (їх номер серед колонок) і передати при ініціації екземпляра класу `SMOTENC(..., categorical_features=cat_feature_indeces)`.
  
  Ви також можете розглянути варіант використання варіації SMOTE, який працює ЛИШЕ з категоріальними ознаками [SMOTEN](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTEN.html)
"""

"""
There should be some work around done with how you do SMOTE with only numerical features and how you combine them with
"""
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train[exclusively_numeric], y_train)

non_numeric = []
for column in columns_list:
  if column not in exclusively_numeric:
    non_numeric.append(column)

non_numeric

"""
Since I took only the numerical features for SMOTE oversampling, the categorical features were left behind. (I lost A HUGE portion of input features).
I need to concatenate them back.
"""

# synthetic: oversampled minority class, I added to the X_train n_syntetic rows to balance the data
n_synthetic = len(X_train_smote) - len(X_train)

n_synthetic

"""
Because since we synthetically created, they're created only for numeric features. If I just concat with categorical, I receive NaN.
To avoid this - I fill values with either random sampling or the mode.
"""

X_train_smote_full_random = pd.concat([
    pd.DataFrame(X_train_smote, columns=exclusively_numeric),
    pd.concat([
        X_train[non_numeric].reset_index(drop=True),
        X_train[non_numeric].sample(n=n_synthetic, replace=True, random_state=0).reset_index(drop=True)
    ], ignore_index=True)
], axis=1)

X_train_smote_full_mode = pd.concat([
    pd.DataFrame(X_train_smote, columns=exclusively_numeric),
    pd.concat([
        X_train[non_numeric].reset_index(drop=True),
        pd.DataFrame(
            [X_train[non_numeric].mode().iloc[0]] * n_synthetic
        ).reset_index(drop=True)
    ], ignore_index=True)
], axis=1)

X_train_smote_full_mode # SMOTE oversampled, will be used for training

X_train_smote_full_random

# oversampling with SMOTENC on all features with SMOTENC
"""
SMOTENC has attribute 'CATEGORICAL_FEATURES' = you should pass the index of categorical features, not the list of features
"""
smotenc = SMOTENC(categorical_features=columns_category_indices, random_state=42)
X_train_smotenc, y_train_smotenc = smotenc.fit_resample(X_train, y_train)

# combining oversampling + undersampling with Smote-tomek
smotetomek = SMOTETomek(random_state=0)
X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)

"""**Завдання 3**.
  1. Навчіть модель логістичної регресії з використанням стратегії One-vs-Rest з логістичною регресією на оригінальних даних, збалансованих з SMOTE, збалансованих з Smote-Tomek.  
  2. Виміряйте якість кожної з натренованих моделей використовуючи `sklearn.metrics.classification_report`.
  3. Напишіть, яку метрику ви обрали для порівняння моделей.
  4. Яка модель найкраща?
  5. Якщо немає суттєвої різниці між моделями - напишіть свою гіпотезу, чому?
"""

# Logistic Regression with strategy = One vs. Rest
log_reg = LogisticRegression(solver="liblinear")
ovr_model = OneVsRestClassifier(log_reg)

# SMOTE training set: NaN replaced with mode
ovr_model.fit(X_train_smote_full_mode, y_train_smote)
ovr_predictions_smote = ovr_model.predict(X_test)
print(classification_report(y_test, ovr_predictions_smote))

# SMOTE training set: NaN replaced with random sampling
ovr_model.fit(X_train_smote_full_random, y_train_smote)
ovr_predictions_smote = ovr_model.predict(X_test)
print(classification_report(y_test, ovr_predictions_smote))

# SMOTENC training set
ovr_model.fit(X_train_smotenc, y_train_smotenc)
ovr_predictions_smotenc = ovr_model.predict(X_test)
print(classification_report(y_test, ovr_predictions_smotenc))

# SMOTE-Tomek traning set
ovr_model.fit(X_train_smotetomek, y_train_smotetomek)
ovr_predictions_smotetomek = ovr_model.predict(X_test)
print(classification_report(y_test, ovr_predictions_smotetomek))

# No resampling techniques applied
ovr_model.fit(X_train, y_train)
ovr_predictions = ovr_model.predict(X_test)
print(classification_report(y_test, ovr_predictions))

"""ANALYSIS OF RESULTS
I CHOSE recall, precision and accuracy to determine the performace. So I get the idea which classes miss the right labels and what is the overall picture for each model.

1. SMOTE oversampling doesn't show good results.
(It doesn't matter if it was mode or random sampling for replacing NaN. The mode worked by 10% better in accuracy than random sampling.)
Overall, SMOTE perfomed the worst. (even with no resampling I got better results.)
It's super bad at identifying classes 1 and 2. In all models, usually the recall, f1-score and presicion are quite low for class 1 and 2.
It has high recall for classes 2 and 3,
It has 35% accuracy which is bad result. I conclude, SMOTE is a bad choice for dataset where there is a combination of categorical and numeric data.
Combining oversampled numeric features with categorical inputs introduces noise and that what affects performance on predicting minority classes.

2. SMOTENC has the best accuracy out of all models(only beats by 1% (SMOTE-Tomek and no resampling applied model)).
The model learns something, but the accuracy is not high.
3. SMOTE-Tomek has a huge gap for predicting class 1. It has a recall of 0.23, which means it doesn't predict correctly in 67% cases.
Overall, the models are better at predicting majority classes, so it still didn't solve the problem.

I suppose, there is some difference in performance between resampling methods: SMOTE didn't work great with categorical data
and SMOTE-Tomek failed to predict class 1 most of the time. SMOTENC showed a good balance between accuracy and predicring each class.
All these methods just have the difference in how great they handle categorical data. So which metric to choose will depend on the input data.

"""