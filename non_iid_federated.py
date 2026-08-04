import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Load the existing CubeSat Dataset
print("Loading CubeSat Structural Dataset...")
data = pd.read_csv('CubeSat_Structural_Dataset.csv')

# 2. Global Train/Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 3. Simulate Non-IID Data by sorting based on structural thickness
print("Sorting data to simulate Non-IID clients (Thin, Medium, and Thick structures)...")
train_data_sorted = train_data.sort_values(by='Thickness_mm')

X_train_full = train_data_sorted[['Thickness_mm', 'Hole_Radius_mm', 'Load_Force_N']].values
y_train_full = train_data_sorted['Max_Von_Mises_Stress_MPa'].values

X_test = test_data[['Thickness_mm', 'Hole_Radius_mm', 'Load_Force_N']].values
y_test = test_data['Max_Von_Mises_Stress_MPa'].values

# Standardize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_full)
y_train_scaled = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).flatten()
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 4. Partition Non-IID Data to 3 Clients
num_clients = 3
split_size = len(X_train_scaled) // num_clients

clients_X = [X_train_scaled[i * split_size: (i + 1) * split_size] for i in range(num_clients)]
clients_y = [y_train_scaled[i * split_size: (i + 1) * split_size] for i in range(num_clients)]

# 5. Centralized Baseline Model
print("Training Centralized Baseline Model...")
centralized_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
centralized_model.fit(X_train_scaled, y_train_scaled)
y_pred_cent = centralized_model.predict(X_test_scaled)
rmse_cent = np.sqrt(mean_squared_error(y_test_scaled, y_pred_cent))

# 6. Federated Learning - Non-IID Scenario
print("Starting Federated Learning (FedAvg) on Non-IID Data...")
global_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1, warm_start=True, random_state=42)
global_model.fit(X_train_scaled[:10], y_train_scaled[:10])

n_rounds = 15
for round_num in range(n_rounds):
    local_coefs = []
    local_intercepts = []

    for i in range(num_clients):
        # Edge Computing: Local Training
        local_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=20, warm_start=True, random_state=42)
        local_model.fit(clients_X[i][:10], clients_y[i][:10])

        local_model.coefs_ = [np.copy(w) for w in global_model.coefs_]
        local_model.intercepts_ = [np.copy(b) for b in global_model.intercepts_]

        local_model.fit(clients_X[i], clients_y[i])

        local_coefs.append(local_model.coefs_)
        local_intercepts.append(local_model.intercepts_)

    # Global Aggregation (FedAvg)
    avg_coefs = [np.mean([local_coefs[client][layer] for client in range(num_clients)], axis=0)
                 for layer in range(len(global_model.coefs_))]
    avg_intercepts = [np.mean([local_intercepts[client][layer] for client in range(num_clients)], axis=0)
                      for layer in range(len(global_model.intercepts_))]

    global_model.coefs_ = avg_coefs
    global_model.intercepts_ = avg_intercepts

y_pred_fed = global_model.predict(X_test_scaled)
rmse_fed = np.sqrt(mean_squared_error(y_test_scaled, y_pred_fed))

print("=" * 50)
print(f"Non-IID Centralized Model RMSE: {rmse_cent:.4f}")
print(f"Non-IID Federated Model RMSE: {rmse_fed:.4f}")
print("=" * 50)

# 7. Visualization
labels = ['Centralized Baseline\n(Unsafe)', 'Federated Learning\n(Secure - Non-IID)']
rmses = [rmse_cent, rmse_fed]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rmses, color=['#d9534f', '#f0ad4e'], width=0.4)
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Validation Performance in Non-IID Scenario\n(Data Heterogeneity)')
plt.ylim(0, max(rmses) * 1.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(rmses) * 0.02), f"{yval:.4f}", ha='center',
             fontweight='bold', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('non_iid_results_chart.png', dpi=300)
print("Chart successfully saved as 'non_iid_results_chart.png'")
plt.show()