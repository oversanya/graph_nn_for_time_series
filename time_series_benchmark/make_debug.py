import numpy as np

# Загрузка исходной матрицы
full_data = np.load('time_series_benchmark/data/PEMS08.npy')  # Предполагаем shape (num_timesteps, num_nodes, num_features)

# Параметры для урезанной версии
debug_num_timesteps = 7 * 24 * 12  # Оставляем 1 день данных

# Создаем уменьшенную версию
print("Original shape:", full_data.shape)
debug_data = full_data[:debug_num_timesteps, :]
print("Debug shape:", debug_data.shape)
# Сохраняем для дебагга
np.save('PEMS08_debug.npy', debug_data)