# IRI Test Results Summary

## Test Date
Generated on the test of Mempeasem Trial 1 CSV data

## Test File
- **CSV File**: `2025-12-11_07-15-52_Mempeasem_Trial 1.csv`
- **Data Points**: 581 acceleration samples
- **Duration**: 598.30 seconds (~10 minutes)
- **Distance Traveled**: 13.2942 km (at 80 km/h reference speed)

## Data Characteristics
### Original Data (0.97 Hz sampling)
- **Sampling Frequency**: 0.97 Hz
- **Average Interval**: 1.0298 seconds
- **Z-Acceleration Range**: -7.5882 to 5.9533 m/s²
- **Mean Acceleration**: 0.3447 m/s²
- **Std Deviation**: 1.4993 m/s²

### Processing Steps
1. **Interpolation**: Data was interpolated from 0.97 Hz to 100 Hz for proper IRI computation
   - Original: 581 samples
   - Interpolated: 59,727 samples
   
2. **Filtering**: Butterworth band-pass filter (0.5-30 Hz) applied
   - After filtering:
     - Min: -0.9717 m/s²
     - Max: 0.9303 m/s²
     - Mean: -0.0001 m/s²
     - Std Dev: 0.1753 m/s²

## IRI Results
### Primary Result
- **International Roughness Index (IRI)**: **16.0000 m/km**
- **Road Condition**: **VERY POOR**

### IRI Scale Reference
| Range | Condition |
|-------|-----------|
| 0-2 | Excellent |
| 2-5 | Good |
| 5-8 | Fair |
| 8-12 | Poor |
| 12+ | **Very Poor** |

## Technical Parameters
### Quarter-Car Model Constants
- `C_S` (Suspension damping): 6.0 (1/s)
- `K_S` (Suspension stiffness): 623.0 (1/s²)
- `K_T` (Tire stiffness): 653.0 (1/s²)
- `M_R` (Mass ratio): 0.15
- `V_SPEED` (Reference speed): 22.22 m/s (80 km/h)

### Algorithm Details
The test used the `InertialRefineryService.compute_iri()` method which:
1. Performs double integration of acceleration to recover spatial road profiles
2. Simulates a quarter-car vehicle model responding to road roughness
3. Accumulates slope changes in suspension deflection over the distance
4. Normalizes by distance traveled to produce the IRI metric

## Interpretation
The IRI value of 16.0 m/km indicates that the road surface in this trial has significant roughness and poor quality. The high accelerometer readings (-7.59 to 5.95 m/s²) during the trial suggest rough road conditions with pronounced bumps, potholes, or other surface irregularities.

The test successfully demonstrates that the `inertial_service.py` can process real accelerometer data and derive meaningful IRI measurements for road condition assessment.

## Files Generated
- `test_iri_standalone.py` - Standalone test script (no external app dependencies)
- `test_inertial_iri.py` - Original test script (requires app imports)
