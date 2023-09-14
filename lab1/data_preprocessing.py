import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Загрузка данных обучающего и тестового
# наборов из файлов CSV для анализа и обработки.
train_data = pd.read_csv('train/train_data.csv')
test_data = pd.read_csv('test/test_data.csv')


# Выполняем стандартизацию признаков
# "temperature" (температура), "humidity" (влажность) и "pressure" (давление)
# в обучающем и тестовом наборах данных.
scaler = StandardScaler()
train_data[['temperature', 'humidity', 'pressure']] = scaler.fit_transform(train_data[['temperature', 'humidity', 'pressure']])
test_data[['temperature', 'humidity', 'pressure']] = scaler.transform(test_data[['temperature', 'humidity', 'pressure']])


# Сохраненяем предобработанные данные обучающего
# и тестового наборов в формате CSV
train_data.to_csv(
    'train/train_data_preprocessed.csv', index=False
    )
test_data.to_csv(
    'test/test_data_preprocessed.csv', index=False
    )
