import numpy as np
import pandas as pd

# Define number of samples for simulation
n_samples = 500

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic mechanical parameters for CubeSat top plate
thickness = np.random.uniform(1.0, 3.5, n_samples)      # Thickness in mm
hole_radius = np.random.uniform(2.0, 5.0, n_samples)    # Hole radius in mm
load_force = np.random.uniform(1000, 5000, n_samples)   # Launch load force in Newtons

# Physical approximation based on solid mechanics and plate bending theory
max_stress = (load_force * 15.0) / (thickness ** 1.8) + (hole_radius * 12.0) + np.random.normal(0, 5.0, n_samples)
max_deflection = (load_force * 2.5) / (thickness ** 3.0) + np.random.normal(0, 0.1, n_samples)

# Create final DataFrame
dataset = pd.DataFrame({
    'Thickness_mm': thickness,
    'Hole_Radius_mm': hole_radius,
    'Load_Force_N': load_force,
    'Max_Von_Mises_Stress_MPa': max_stress,
    'Max_Deflection_mm': max_deflection
})

# Save dataset to CSV file
dataset.to_csv('CubeSat_Structural_Dataset.csv', index=False)

print("Dataset successfully generated and saved as CubeSat_Structural_Dataset.csv")
print(dataset.head())