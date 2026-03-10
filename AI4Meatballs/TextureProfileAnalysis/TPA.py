"""
TPA Analysis
Skyler St. Pierre, Lucas Boyle, Aeneas Koosis
Last updated March 4, 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import stats


def read_data_file(file_path, sheet_name=None):
    if file_path.endswith(('.xlsx', '.xls')):
        if sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            return pd.read_excel(file_path)
    else:
        # Strip trailing tabs from meatball txt files
        with open(file_path, 'r') as f:
            lines = [line.rstrip('\t\n\r ') for line in f]
        from io import StringIO
        return pd.read_csv(StringIO('\n'.join(lines)), sep="\t")


radius = 4
surf = radius * radius * np.pi


def find_exceed_under(arr, half, threshold=0.1):
    first_exceed_idx = next((i for i in range(len(arr)) if arr[i] > threshold), -1)
    max_first_half_idx = max(range(half), key=lambda i: arr[i], default=-1)

    first_under_idx = -1
    if max_first_half_idx != -1:
        first_under_idx = next((i for i in range(max_first_half_idx + 1, half) if arr[i] < threshold), -1)

    max_second_half_idx = max(range(half, len(arr)), key=lambda i: arr[i], default=-1)

    second_under_idx = -1
    if max_second_half_idx != -1:
        second_under_idx = next((i for i in range(max_second_half_idx + 1, len(arr)) if arr[i] < threshold), -1)

    second_exceed_idx = -1
    if first_under_idx != -1:
        second_exceed_idx = next((i for i in range(half, len(arr)) if arr[i] > threshold), -1)

    return first_exceed_idx, first_under_idx, second_exceed_idx, second_under_idx


def find_max_index(arr):
    return np.argmax(arr)


def positive_area_under_curve(force, index1, index2):
    force_section = force[index1:index2 + 1]
    force_section = np.where(force_section > 0, force_section, 0)
    area = np.trapezoid(force_section)
    return float(area)


def find_start_end(gap):
    gap = np.array(gap)
    start = next((i for i in range(1, len(gap)) if abs(gap[i] - gap[i - 1]) > 0.5), 0)
    end = next((i for i in range(len(gap) - 1, 0, -1) if abs(gap[i] - gap[i - 1]) > 0.5), len(gap) - 1)
    second_end = next((i for i in range(end - 1, 0, -1) if abs(gap[i] - gap[i - 1]) > 0.5), end)
    return start, second_end


def double_compression_data(file, plotting=False, gap_filter=False, force_filter=True, sig=2):
    file_path = file
    df = read_data_file(file_path)
    numeric_data = df.apply(pd.to_numeric, errors='coerce').to_numpy()

    if ".txt" in file:
        gap = numeric_data[:, 5]
        force = numeric_data[:, 4]
    else:
        gap = numeric_data[:, 5]
        force = numeric_data[:, 4]

    time = df.index.to_numpy()

    if force_filter:
        force = gaussian_filter1d(force, sigma=sig)

    if gap_filter:
        force = gaussian_filter1d(gap, sigma=sig)

    start, end = find_start_end(gap)
    half = start + (end - start) / 2

    first_exceed_idx, first_under_idx, second_exceed_idx, second_under_idx = find_exceed_under(force, int(half),
                                                                                               threshold=0.1)

    force1 = force[first_exceed_idx: first_under_idx]
    force2 = force[second_exceed_idx: second_under_idx]
    max_index1 = int(find_max_index(force1))
    max_index2 = int(find_max_index(force2))

    A1 = positive_area_under_curve(force, first_exceed_idx, first_exceed_idx + max_index1)
    A2 = positive_area_under_curve(force, first_exceed_idx + max_index1, first_under_idx)
    A3 = positive_area_under_curve(force, second_exceed_idx, second_exceed_idx + max_index2)
    A4 = positive_area_under_curve(force, second_exceed_idx + max_index2, second_under_idx)

    t1 = float(time[first_exceed_idx + max_index1] - time[first_exceed_idx])
    t2 = float(time[second_exceed_idx + max_index2] - time[second_exceed_idx])

    F1 = float(force[first_exceed_idx + max_index1])
    F2 = float(force[second_exceed_idx + max_index2])

    strain = 0.5
    stiffness = (F1 / surf) / strain * 1000
    hardness = F1
    cohesiveness = (A3 + A4) / (A1 + A2)
    springiness = t2 / t1
    resilience = A2 / A1
    chewiness = F1 * (A3 + A4) / (A1 + A2) * t2 / t1

    if plotting:
        fig, ax1 = plt.subplots()
        # Plot Gap on the primary y-axis
        ax1.plot(gap, color='red', linewidth=2, label='Gap [μm]')
        ax1.set_xlabel("index")  # Adjust x-label as needed
        ax1.set_ylabel("gap [μm]", color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.invert_yaxis()
        ax1.set_xlim(start, end)
        # Create a secondary y-axis for Force
        ax2 = ax1.twinx()
        ax2.plot(force, color='black', linewidth=2, label='Force [N]')
        ax2.set_ylabel("force [N]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        plt.show()

    return stiffness, hardness, cohesiveness, springiness, resilience, chewiness


def compute_tpa_stats(filenames):
    stiffness_vals = []
    hardness_vals = []
    cohesiveness_vals = []
    springiness_vals = []
    resilience_vals = []
    chewiness_vals = []

    for filename in filenames:
        try:
            stiffness, hardness, cohesiveness, springiness, resilience, chewiness = double_compression_data(filename)
            stiffness_vals.append(stiffness)
            hardness_vals.append(hardness)
            cohesiveness_vals.append(cohesiveness)
            springiness_vals.append(springiness)
            resilience_vals.append(resilience)
            chewiness_vals.append(chewiness)
        except Exception as e:
            print(f"  Error: {os.path.basename(filename)} - {e}")

    stats_dict = {
        "stiffness": (np.mean(stiffness_vals), np.std(stiffness_vals)),
        "hardness": (np.mean(hardness_vals), np.std(hardness_vals)),
        "cohesiveness": (np.mean(cohesiveness_vals), np.std(cohesiveness_vals)),
        "springiness": (np.mean(springiness_vals), np.std(springiness_vals)),
        "resilience": (np.mean(resilience_vals), np.std(resilience_vals)),
        "chewiness": (np.mean(chewiness_vals), np.std(chewiness_vals))
    }

    return stats_dict, cohesiveness_vals


# ============= CHANGE: FILE PATHS FOR MEATBALL DATA =============

base_dir = "/Users/Your/FilePath"

# Get file lists (excluding _fail samples)
beef_filenames = [f for f in [os.path.join(base_dir, "100pBeef", f"100pBeef{i}.txt") for i in range(1, 15)]
                  if os.path.exists(f) and '_fail' not in f]
blended_filenames = [os.path.join(base_dir, "BlendMushBeef", f"Blend{i}.txt") for i in range(1, 11)]
gardein_filenames = [os.path.join(base_dir, "Gardein", f"Gardein{i}.txt") for i in range(1, 11)]
impossible_filenames = [os.path.join(base_dir, "Impossible", f"Impossible{i}.txt") for i in range(1, 11)]

# Filter to existing files only
beef_filenames = [f for f in beef_filenames if os.path.exists(f)]
blended_filenames = [f for f in blended_filenames if os.path.exists(f)]
gardein_filenames = [f for f in gardein_filenames if os.path.exists(f)]
impossible_filenames = [f for f in impossible_filenames if os.path.exists(f)]

print(beef_filenames)

print("=" * 80)
print("TPA ANALYSIS")
print("=" * 80)

# Compute stats
print("\nProcessing 100% Beef...")
beef_stats, beef_coh = compute_tpa_stats(beef_filenames)
print(f"  n = {len(beef_coh)}")

print("\nProcessing Blended...")
blended_stats, blended_coh = compute_tpa_stats(blended_filenames)
print(f"  n = {len(blended_coh)}")

print("\nProcessing Gardein...")
gardein_stats, gardein_coh = compute_tpa_stats(gardein_filenames)
print(f"  n = {len(gardein_coh)}")

print("\nProcessing Impossible...")
impossible_stats, impossible_coh = compute_tpa_stats(impossible_filenames)
print(f"  n = {len(impossible_coh)}")

# Print results
datasets = {
    "100% Beef": beef_stats,
    "Blended": blended_stats,
    "Gardein": gardein_stats,
    "Impossible": impossible_stats
}

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

metrics = ["stiffness", "hardness", "cohesiveness", "springiness", "resilience", "chewiness"]
units = ["[kPa]", "[N]", "[-]", "[-]", "[-]", "[N]"]

for idx, metric in enumerate(metrics):
    print(f"\n{metric.upper()} {units[idx]}:")
    for name, stats_dict in datasets.items():
        mean, std = stats_dict[metric]
        print(f"  {name:<12}: {mean:.3f} +/- {std:.3f}")

# Print cohesiveness summary
print("\n" + "=" * 80)
print("COHESIVENESS SUMMARY")
print("=" * 80)
print(f"100% Beef:  {beef_stats['cohesiveness'][0]:.3f} +/- {beef_stats['cohesiveness'][1]:.3f}")
print(f"Blended:    {blended_stats['cohesiveness'][0]:.3f} +/- {blended_stats['cohesiveness'][1]:.3f}")
print(f"Gardein:    {gardein_stats['cohesiveness'][0]:.3f} +/- {gardein_stats['cohesiveness'][1]:.3f}")
print(f"Impossible: {impossible_stats['cohesiveness'][0]:.3f} +/- {impossible_stats['cohesiveness'][1]:.3f}")


## Plotting ##
def double_compression_process(file, plotting=False, gap_filter=False, force_filter=True, sig=2):
    file_path = file
    df = read_data_file(file_path)
    numeric_data = df.apply(pd.to_numeric, errors='coerce').to_numpy()

    # DEFINE COLUMNS FOR GAP AND FORCE
    if ".txt" in file:
        gap = numeric_data[:, 5]
        force = numeric_data[:, 4]
    else:
        gap = numeric_data[:, 5]
        force = numeric_data[:, 4]

    if force_filter:
        force = gaussian_filter1d(force, sigma=sig)

    if gap_filter:
        force = gaussian_filter1d(gap, sigma=sig)

    start, end = find_start_end(gap)
    gap_red = gap[start:end]
    force_red = force[start:end]

    return gap_red, force_red


def process_force_curves(speed_label, color, filenames):
    force_curves = []
    for filename in filenames:
        gap_reduction, force_reduction = double_compression_process(filename)
        force_curves.append(force_reduction)
    # Trim to shortest curve length
    min_length = min(len(curve) for curve in force_curves)
    trimmed_curves = np.array([curve[:min_length] for curve in force_curves])

    # Calculate mean and standard deviation
    mean_force = np.mean(trimmed_curves, axis=0)
    std_force = np.std(trimmed_curves, axis=0)

    # Time axis
    time = np.arange(min_length) / 244  # Time in seconds

    return time, mean_force, std_force, color, speed_label


PureBeef = process_force_curves('25/s', (132/255, 60/255, 12/255, 1.0), beef_filenames) 
Blended = process_force_curves('25/s', (132/255, 60/255, 12/255, 1.0), blended_filenames)  
Gardein = process_force_curves('25/s', (132/255, 60/255, 12/255, 1.0), gardein_filenames)  
Impossible = process_force_curves('25/s', (132/255, 60/255, 12/255, 1.0), impossible_filenames)  

# CHANGE VARIABLE NAMES AND COLORS TO MATCH DATA
# Combined Plots for Each Speed with Blue and Red
TrueRed = '#FF0000'  # 100% beef
DarkRed = '#990000'  # blended
TrueBlue = '#0080FF'  # impossible
DarkBlue = '#004C99'  # gardein
fig, ax = plt.subplots(figsize=(6.2, 4), dpi=300)
for (data, label, color) in [
    (PureBeef, "100pBeef (n=8)", TrueRed),
    (Blended, "Blended (n=9)", DarkRed),
    (Gardein, "Gardein (n=10)", DarkBlue),
    (Impossible, "Impossible (n=10)", TrueBlue),
]:
    # Plot inplane data in red
    time, mean_force, std_force, old_color, old_label = data
    ax.plot(time, mean_force, color=color, linestyle='-', linewidth=3, label=f'{label}')
    plt.fill_between(time, mean_force - std_force, mean_force + std_force, color=color, alpha=0.3)


plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.legend()
plt.title("Double Compression - 25%/s")
plt.xlim(left=0, right=8)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('TPA_mean_STD_plant')
plt.show()
