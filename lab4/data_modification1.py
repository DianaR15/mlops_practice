import pandas as pd


# Загрузка датасета
df = pd.read_csv('datasets/data.csv')

# Оставление только необходимых столбцов
df = df[['Pclass', 'Sex', 'Age']] 

# Сохранение модифицированного датасета
df.to_csv('datasets/data_mod1.csv', index=False)
