# app/core/config.py
import numpy as np
from scipy.interpolate import PchipInterpolator

# Golden Quarter-Car Simulation (QCM) Constants
C_S = 6.0  # Suspension damping coefficient (1/s)
K_S = 623.0  # Suspension stiffness coefficient (1/s^2)
K_T = 653.0  # Tire stiffness coefficient (1/s^2)
M_R = 0.15  # Mass ratio (Unsprung mass / Sprung mass)
V_SPEED = 22.22  # Standard calibration velocity (80 km/h in m/s)
V_REF_KMH = 80.0  # Standard reference speed for IRI computation (km/h)

# IRI Segment Configuration
IRI_SEGMENT_LENGTH_M = 50.0  # Standard segment length for IRI reporting (m)

# Parametric PCHIP Spline Control Knots
MATRIX_KNOTS_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

PCHIP_ALLIGATOR = PchipInterpolator(MATRIX_KNOTS_X, np.array([5.0, 12.0, 22.0, 30.0, 45.0, 60.0, 72.0, 85.0, 100.0]))
PCHIP_LINEAR = PchipInterpolator(MATRIX_KNOTS_X, np.array([2.0, 6.0, 12.0, 18.0, 28.0, 42.0, 50.0, 68.0, 85.0]))
PCHIP_POTHOLE = PchipInterpolator(
    MATRIX_KNOTS_X,
    np.array([15.0, 28.0, 45.0, 55.0, 70.0, 85.0, 90.0, 98.0, 100.0]),
)
