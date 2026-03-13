#!/usr/bin/env python3
"""
Test the new chart functions with dummy data.
Run from f1_webapp directory.
"""
import sys
sys.path.insert(0, '..')

from components.charts import (
    sector_performance_radar,
    sector_correlation_matrix,
    sector_evolution_analysis
)
import plotly.graph_objects as go

# Create dummy sector_stats
sector_stats = {
    1: {
        "sector_stats": {
            "sector1": {"mean": 25.5, "std": 0.3},
            "sector2": {"mean": 30.2, "std": 0.4},
            "sector3": {"mean": 28.7, "std": 0.2},
        },
        "contribution_stats": {
            "sector1_pct_mean": 33.1,
            "sector2_pct_mean": 33.5,
            "sector3_pct_mean": 33.4,
        }
    },
    2: {
        "sector_stats": {
            "sector1": {"mean": 25.8, "std": 0.4},
            "sector2": {"mean": 30.5, "std": 0.3},
            "sector3": {"mean": 29.0, "std": 0.5},
        },
        "contribution_stats": {
            "sector1_pct_mean": 33.2,
            "sector2_pct_mean": 33.3,
            "sector3_pct_mean": 33.5,
        }
    },
    3: {
        "sector_stats": {
            "sector1": {"mean": 25.3, "std": 0.2},
            "sector2": {"mean": 30.0, "std": 0.5},
            "sector3": {"mean": 28.5, "std": 0.3},
        },
        "contribution_stats": {
            "sector1_pct_mean": 33.0,
            "sector2_pct_mean": 33.6,
            "sector3_pct_mean": 33.4,
        }
    },
}

# Create dummy driver_info
driver_info = [
    {"driver_number": 1, "full_name": "Driver A", "team_colour": "#FF0000", "team_name": "Team Red"},
    {"driver_number": 2, "full_name": "Driver B", "team_colour": "#00FF00", "team_name": "Team Green"},
    {"driver_number": 3, "full_name": "Driver C", "team_colour": "#0000FF", "team_name": "Team Blue"},
]

# Create dummy valid_laps for evolution analysis
valid_laps = []
for lap_num in range(1, 6):
    for drv_num in [1, 2, 3]:
        valid_laps.append({
            "driver_number": drv_num,
            "lap_number": lap_num,
            "duration_sector_1": 25.5 + (drv_num * 0.1) + (lap_num * 0.05),
            "duration_sector_2": 30.2 + (drv_num * 0.15) - (lap_num * 0.02),
            "duration_sector_3": 28.7 + (drv_num * 0.05) + (lap_num * 0.01),
        })

print("Testing sector_performance_radar...")
try:
    fig1 = sector_performance_radar(sector_stats, driver_info)
    print(f"  ✅ Figure created: {type(fig1)}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting sector_correlation_matrix...")
try:
    fig2 = sector_correlation_matrix(sector_stats, driver_info)
    print(f"  ✅ Figure created: {type(fig2)}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting sector_evolution_analysis...")
try:
    fig3 = sector_evolution_analysis(valid_laps, sector_stats, driver_info)
    print(f"  ✅ Figure created: {type(fig3)}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test complete ---")