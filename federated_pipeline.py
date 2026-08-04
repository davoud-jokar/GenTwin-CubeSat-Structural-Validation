import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load Data and Preprocessing
print("Loading CubeSat Structural Dataset...")
data = pd.read_csv('CubeSat_Structural_Dataset.csv')

# Inputs: Thickness, Hole Radius, Load Force | Output: Max Von Mises Stress
X = data[['Thickness_mm', 'Hole_Radius_mm', 'Load_Force_N']].values
y = data['Max_Von_Mises_Stress_MPa'].values

# Standardizing Data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 2. Simulate Decentralized Startups (Data Partitioning)
num_clients = 3
split_size = len(X_train) // num_clients
clients_X = [X_train[i * split_size: (i + 1) * split_size] for i in range(num_clients)]
clients_y = [y_train[i * split_size: (i + 1) * split_size] for i in range(num_clients)]

# 3. Centralized Baseline (Unsafe/High IP Risk)
print("Training Centralized Baseline Model...")
centralized_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
centralized_model.fit(X_train, y_train)
y_pred_cent = centralized_model.predict(X_test)
rmse_cent = np.sqrt(mean_squared_error(y_test, y_pred_cent))

# 4. Federated Learning (GenTwin Architecture) - Secure
print("Starting Federated Learning (FedAvg) rounds...")

# Initialize global model
global_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1, warm_start=True, random_state=42)
global_model.fit(X_train[:10], y_train[:10])  # Proper initialization of layers

n_rounds = 15
for round_num in range(n_rounds):
    local_coefs = []
    local_intercepts = []

    for i in range(num_clients):
        # Local Training on Edge Nodes
        local_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=20, warm_start=True, random_state=42)
        # Initialize with full data to set structure
        local_model.fit(clients_X[i][:10], clients_y[i][:10])

        # Load weights from global model
        local_model.coefs_ = [np.copy(w) for w in global_model.coefs_]
        local_model.intercepts_ = [np.copy(b) for b in global_model.intercepts_]

        # Train on local startup dataset
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

y_pred_fed = global_model.predict(X_test)
rmse_fed = np.sqrt(mean_squared_error(y_test, y_pred_fed))

print("=" * 50)
print(f"Centralized Model RMSE: {rmse_cent:.4f}")
print(f"GenTwin Federated Model RMSE: {rmse_fed:.4f}")
print("=" * 50)

# 5. Result Visualization
labels = ['Centralized Baseline\n(Unsafe)', 'GenTwin Federated\n(Secure)']
rmses = [rmse_cent, rmse_fed]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rmses, color=['#d9534f', '#5cb85c'], width=0.4)
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Validation Performance: Centralized vs. Federated Architecture\n(CubeSat Structural Dataset)')
plt.ylim(0, max(rmses) * 1.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(rmses) * 0.02), f"{yval:.4f}", ha='center',
             fontweight='bold', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('experimental_results_chart.png', dpi=300)
print("Chart successfully saved as 'experimental_results_chart.png'")
plt.show()