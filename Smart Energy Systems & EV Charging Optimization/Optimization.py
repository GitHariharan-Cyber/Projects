import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# SMART ENERGY SYSTEMS & EV CHARGING OPTIMIZATION
# ============================================================
# Features:
# 1. Time-series residential demand generation
# 2. Dynamic electricity pricing
# 3. SOC-aware EV battery charging model
# 4. Charging efficiency variation with SOC
# 5. Cheapest-hour charging optimizer
# 6. Grid-aware load scheduling
# 7. Real-time charging scheduler
# 8. Visualization and cost comparison
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BatteryConfig:
    capacity_kwh: float = 60.0
    soc_initial: float = 0.20
    soc_target: float = 0.90
    max_charging_power_kw: float = 11.0
    charging_efficiency_low_soc: float = 0.96
    charging_efficiency_high_soc: float = 0.88


@dataclass
class SimulationConfig:
    timestep_minutes: int = 60
    total_hours: int = 24
    base_load_mean_kw: float = 2.5
    base_load_std_kw: float = 0.8


# ============================================================
# DYNAMIC ELECTRICITY PRICE MODEL
# ============================================================

def generate_dynamic_price_profile() -> np.ndarray:
    """
    Simulated time-of-use electricity pricing.
    Units: EUR/kWh
    """

    price = np.array([
        0.18, 0.17, 0.16, 0.16,
        0.17, 0.20, 0.24, 0.28,
        0.31, 0.29, 0.26, 0.25,
        0.24, 0.23, 0.22, 0.24,
        0.27, 0.33, 0.36, 0.34,
        0.30, 0.26, 0.22, 0.19
    ])

    return price


# ============================================================
# STOCHASTIC DEMAND MODEL
# ============================================================

def generate_residential_load(config: SimulationConfig) -> np.ndarray:
    """
    Generate stochastic residential demand profile.
    """

    np.random.seed(42)

    base_profile = np.array([
        1.5, 1.4, 1.3, 1.3,
        1.5, 2.0, 3.2, 4.0,
        3.8, 3.2, 2.8, 2.5,
        2.4, 2.5, 2.7, 3.0,
        3.8, 5.0, 5.5, 5.2,
        4.0, 3.0, 2.2, 1.8
    ])

    noise = np.random.normal(
        loc=0,
        scale=config.base_load_std_kw,
        size=config.total_hours
    )

    stochastic_load = base_profile + noise

    stochastic_load = np.clip(stochastic_load, 0.5, None)

    return stochastic_load


# ============================================================
# SOC-DEPENDENT EFFICIENCY MODEL
# ============================================================

def charging_efficiency(soc: float,
                        battery: BatteryConfig) -> float:
    """
    Charging efficiency decreases at high SOC.
    """

    if soc < 0.8:
        return battery.charging_efficiency_low_soc

    reduction = (soc - 0.8) / 0.2

    eff = (
        battery.charging_efficiency_low_soc
        - reduction * (
            battery.charging_efficiency_low_soc
            - battery.charging_efficiency_high_soc
        )
    )

    return max(eff, battery.charging_efficiency_high_soc)


# ============================================================
# BATTERY CHARGING MODEL
# ============================================================

def simulate_ev_charging(
    battery: BatteryConfig,
    charging_schedule: np.ndarray,
    electricity_price: np.ndarray
):
    """
    Simulate EV charging with SOC-aware dynamics.
    """

    dt_hours = 1.0

    soc = battery.soc_initial

    soc_history = []
    power_history = []
    cost_history = []

    total_cost = 0

    for hour in range(24):

        requested_power = charging_schedule[hour]

        if soc >= battery.soc_target:
            actual_power = 0
        else:
            # Hardware charging limit reduces near full SOC
            soc_limit_factor = max(0.15, 1 - soc)

            hardware_limit = (
                battery.max_charging_power_kw
                * soc_limit_factor
            )

            actual_power = min(
                requested_power,
                hardware_limit
            )

        eff = charging_efficiency(soc, battery)

        energy_added = actual_power * eff * dt_hours

        delta_soc = energy_added / battery.capacity_kwh

        soc = min(soc + delta_soc, 1.0)

        charging_cost = (
            actual_power
            * electricity_price[hour]
            * dt_hours
        )

        total_cost += charging_cost

        soc_history.append(soc)
        power_history.append(actual_power)
        cost_history.append(charging_cost)

    return {
        'soc_history': np.array(soc_history),
        'power_history': np.array(power_history),
        'cost_history': np.array(cost_history),
        'total_cost': total_cost,
        'final_soc': soc
    }


# ============================================================
# COST OPTIMIZER
# ============================================================

def optimize_charging_schedule(
    battery: BatteryConfig,
    electricity_price: np.ndarray,
    base_load: np.ndarray,
    grid_limit_kw: float = 12.0
):
    """
    Cheapest-hour optimizer with grid-awareness.
    """

    energy_needed = (
        battery.soc_target
        - battery.soc_initial
    ) * battery.capacity_kwh

    estimated_hours = int(np.ceil(
        energy_needed / battery.max_charging_power_kw
    ))

    cheapest_hours = np.argsort(electricity_price)

    charging_schedule = np.zeros(24)

    remaining_energy = energy_needed

    for hour in cheapest_hours:

        available_grid_capacity = (
            grid_limit_kw - base_load[hour]
        )

        available_grid_capacity = max(
            available_grid_capacity,
            0
        )

        charging_power = min(
            battery.max_charging_power_kw,
            available_grid_capacity
        )

        if charging_power <= 0:
            continue

        energy_this_hour = charging_power

        if remaining_energy <= 0:
            break

        if energy_this_hour > remaining_energy:
            charging_power = remaining_energy
            energy_this_hour = remaining_energy

        charging_schedule[hour] = charging_power

        remaining_energy -= energy_this_hour

    return charging_schedule


# ============================================================
# REAL-TIME SCHEDULER
# ============================================================

def scheduler_step(
    current_hour: int,
    charging_schedule: np.ndarray
) -> float:
    """
    Real-time scheduler logic.
    """

    return charging_schedule[current_hour]


# ============================================================
# BASELINE UNCONTROLLED CHARGING
# ============================================================

def uncontrolled_charging_strategy(
    battery: BatteryConfig
):
    """
    Immediate charging without optimization.
    """

    schedule = np.zeros(24)

    energy_needed = (
        battery.soc_target
        - battery.soc_initial
    ) * battery.capacity_kwh

    charging_hours = int(np.ceil(
        energy_needed / battery.max_charging_power_kw
    ))

    start_hour = 18

    for i in range(charging_hours):
        hour = (start_hour + i) % 24
        schedule[hour] = battery.max_charging_power_kw

    return schedule


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_peak_load(
    base_load: np.ndarray,
    charging_power: np.ndarray
):
    return np.max(base_load + charging_power)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(
    electricity_price,
    base_load,
    optimized_results,
    uncontrolled_results,
    optimized_schedule,
    uncontrolled_schedule
):

    hours = np.arange(24)

    plt.figure(figsize=(15, 12))

    # --------------------------------------------------------
    # Electricity Price
    # --------------------------------------------------------
    plt.subplot(4, 1, 1)
    plt.plot(hours, electricity_price, marker='o')
    plt.grid(True)
    plt.ylabel('EUR/kWh')
    plt.title('Dynamic Electricity Price')

    # --------------------------------------------------------
    # Residential Load
    # --------------------------------------------------------
    plt.subplot(4, 1, 2)
    plt.plot(hours, base_load, marker='o')
    plt.grid(True)
    plt.ylabel('kW')
    plt.title('Residential Demand Profile')

    # --------------------------------------------------------
    # Charging Schedule
    # --------------------------------------------------------
    plt.subplot(4, 1, 3)
    plt.step(
        hours,
        optimized_schedule,
        where='mid',
        label='Optimized Charging'
    )

    plt.step(
        hours,
        uncontrolled_schedule,
        where='mid',
        label='Uncontrolled Charging'
    )

    plt.grid(True)
    plt.ylabel('Charging Power (kW)')
    plt.title('Charging Schedule Comparison')
    plt.legend()

    # --------------------------------------------------------
    # SOC Comparison
    # --------------------------------------------------------
    plt.subplot(4, 1, 4)

    plt.plot(
        hours,
        optimized_results['soc_history'] * 100,
        marker='o',
        label='Optimized SOC'
    )

    plt.plot(
        hours,
        uncontrolled_results['soc_history'] * 100,
        marker='s',
        label='Uncontrolled SOC'
    )

    plt.grid(True)
    plt.ylabel('SOC (%)')
    plt.xlabel('Hour of Day')
    plt.title('Battery SOC Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Initialize Configuration
    # --------------------------------------------------------
    battery = BatteryConfig()
    sim_config = SimulationConfig()

    # --------------------------------------------------------
    # Generate Profiles
    # --------------------------------------------------------
    electricity_price = generate_dynamic_price_profile()

    residential_load = generate_residential_load(sim_config)

    # --------------------------------------------------------
    # Optimized Charging
    # --------------------------------------------------------
    optimized_schedule = optimize_charging_schedule(
        battery=battery,
        electricity_price=electricity_price,
        base_load=residential_load,
        grid_limit_kw=12
    )

    optimized_results = simulate_ev_charging(
        battery=battery,
        charging_schedule=optimized_schedule,
        electricity_price=electricity_price
    )

    # --------------------------------------------------------
    # Uncontrolled Charging
    # --------------------------------------------------------
    uncontrolled_schedule = uncontrolled_charging_strategy(
        battery
    )

    uncontrolled_results = simulate_ev_charging(
        battery=battery,
        charging_schedule=uncontrolled_schedule,
        electricity_price=electricity_price
    )

    # --------------------------------------------------------
    # Peak Load Analysis
    # --------------------------------------------------------
    optimized_peak = calculate_peak_load(
        residential_load,
        optimized_results['power_history']
    )

    uncontrolled_peak = calculate_peak_load(
        residential_load,
        uncontrolled_results['power_history']
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    print('\n====================================================')
    print('SMART ENERGY SYSTEMS & EV CHARGING OPTIMIZATION')
    print('====================================================')

    print('\nOptimized Charging Results')
    print('----------------------------')
    print(f'Total Charging Cost: EUR {optimized_results["total_cost"]:.2f}')
    print(f'Final SOC: {optimized_results["final_soc"] * 100:.2f}%')
    print(f'Peak Grid Load: {optimized_peak:.2f} kW')

    print('\nUncontrolled Charging Results')
    print('------------------------------')
    print(f'Total Charging Cost: EUR {uncontrolled_results["total_cost"]:.2f}')
    print(f'Final SOC: {uncontrolled_results["final_soc"] * 100:.2f}%')
    print(f'Peak Grid Load: {uncontrolled_peak:.2f} kW')

    savings = (
        uncontrolled_results['total_cost']
        - optimized_results['total_cost']
    )

    savings_percent = (
        savings / uncontrolled_results['total_cost']
    ) * 100

    print('\nOptimization Impact')
    print('------------------------------')
    print(f'Cost Savings: EUR {savings:.2f}')
    print(f'Cost Reduction: {savings_percent:.2f}%')

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    plot_results(
        electricity_price,
        residential_load,
        optimized_results,
        uncontrolled_results,
        optimized_schedule,
        uncontrolled_schedule
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    main()
