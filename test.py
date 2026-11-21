import pandas as pd
import numpy as np
import re 
import pandapower.networks as pn
import pandapower as pp
from numpy.linalg import pinv
import os


def load_dataset(number_of_rows): 
    df=pd.read_csv('IEEE118Normal_Corrected.csv')
    # select a random sample of rows
    df_sampled = df.sample(n=number_of_rows, random_state=1).reset_index(drop=True)
    return df_sampled


def extract_bus_number(col_name):
    return int(re.findall(r'\d+', col_name)[0])


def compute_net_injection(row,pgpl_columns):
    values = []
    for col in pgpl_columns:
        if col.startswith("PG"):
            values.append(row[col])  # generation → positive injection
        else:
            #values.append(-row[col])  # load → negative injection # OLD
            values.append(row[col])  # load → negative injection
    return values

def xyz(df,buses,attack_mag_range,meter_range,authentic_name,fdia_name):
    # STEP 2: Identify all PG and PL columns
    pg_cols = [col for col in df.columns if col.startswith("PG")]
    pl_cols = [col for col in df.columns if col.startswith("PL")]


    pg_bus_map = [(col, extract_bus_number(col)) for col in pg_cols]
    pl_bus_map = [(col, extract_bus_number(col)) for col in pl_cols]

    # Combine and sort by bus number
    pgpl_combined = pg_bus_map + pl_bus_map
    pgpl_sorted = sorted(pgpl_combined, key=lambda x: x[1])  # sort by bus number

    # Get final ordered list of column names
    pgpl_columns = [col for col, _ in pgpl_sorted]

    Z_inj = np.array(df.apply(compute_net_injection, axis=1, pgpl_columns=pgpl_columns).tolist())

    net = pn.case118()


    slack_bus_idx = net.ext_grid['bus'].values[0]  # usually only one slack bus
    all_bus_idxs = net.bus.index.tolist()
    state_bus_idxs = [bus for bus in all_bus_idxs if bus != slack_bus_idx]

    # Map bus index → θ state index (excluding slack)
    bus_to_state_idx = {bus: idx for idx, bus in enumerate(state_bus_idxs)}  # θ ∈ ℝ^117

    # STEP 3: Extract branch info and compute reactance & susceptance
    branch_from = net.line.from_bus.values
    branch_to = net.line.to_bus.values
    x_per_km = net.line["x_ohm_per_km"].values
    length_km = net.line["length_km"].values

    # Total reactance per line
    branch_x = x_per_km * length_km
    branch_x[branch_x == 0] = 1e-6  # avoid division by zero

    # Susceptance: b_ij = 1 / x_ij
    branch_b = 1 / branch_x

    n_buses = len(net.bus)
    n_branches = len(net.line)
    n_states = len(bus_to_state_idx)  # 117 (excluding slack)
    n_measurements = n_buses + n_branches  # 118 injections + 173 flows = 291

    # STEP 4: Initialize H matrix
    H = np.zeros((n_measurements, n_states))

# STEP 5: Injection rows (bus-wise) — rows 0 to 117
    for i, bus in enumerate(net.bus.index):
        if bus == slack_bus_idx:
            continue  # skip slack bus
        theta_i = bus_to_state_idx[bus]
    
        # Find all lines connected to this bus
        connected_lines = net.line[(net.line.from_bus == bus) | (net.line.to_bus == bus)].index
        for line in connected_lines:
            from_bus = branch_from[line]
            to_bus = branch_to[line]
            b = branch_b[line]

            if from_bus == bus:
                other_bus = to_bus
            else:
                other_bus = from_bus

            if other_bus == slack_bus_idx:
                H[i, theta_i] += b
            elif other_bus in bus_to_state_idx:
                theta_j = bus_to_state_idx[other_bus]
                H[i, theta_i] += b
                H[i, theta_j] -= b


    # STEP 6: Flow rows (line-wise) — rows 118 to 290
    offset = n_buses  # flow rows start after injection rows
    for idx, line in enumerate(net.line.index):
        from_bus = branch_from[idx]
        to_bus = branch_to[idx]
        b = branch_b[idx]
        row = offset + idx
        if from_bus != slack_bus_idx:
            H[row, bus_to_state_idx[from_bus]] += b
        if to_bus != slack_bus_idx:
            H[row, bus_to_state_idx[to_bus]] -= b
    
    # STEP 1: Use only PG/PL injection part of H (first 118 rows)
    H_inj = H[:n_buses, :]               # shape: [118, 117]
    H_inj_pinv = pinv(H_inj)             # shape: [117, 118]

    # STEP 2: Set up dimensions
    n_rows = Z_inj.shape[0]              # number of snapshots
    n_inj = n_buses                      # 118 injections
    n_flow = n_branches                  # 173 flows
    n_meas = n_inj + n_flow              # 291 total measurements

    Z_full = np.zeros((n_rows, n_meas))  # final output matrix
    # Optional reverse map
    state_idx_to_bus = {v: k for k, v in bus_to_state_idx.items()}

    # STEP 3: Loop through all rows and compute z = [injections | flows]
    for i in range(n_rows):
        z_inj = Z_inj[i]  # injection vector from dataset (PG/PL)

    # Estimate voltage angles: θ̂ = pseudo-inverse(H_inj) * z_inj
        theta_hat = H_inj_pinv @ z_inj  # shape: [117]

    # Compute flows: f_ij = b_ij * (θ_i - θ_j)
        flows = []
        for line_idx in range(n_branches):
            from_bus = branch_from[line_idx]
            to_bus = branch_to[line_idx]
            b = branch_b[line_idx]
            theta_i = theta_j = 0.0
            if from_bus != slack_bus_idx:
                theta_i = theta_hat[bus_to_state_idx[from_bus]]
            if to_bus != slack_bus_idx:
                theta_j = theta_hat[bus_to_state_idx[to_bus]]

            flow = b * (theta_i - theta_j)
            flows.append(flow)

    # Assemble full measurement vector: [PG/PL | line_flows]
        Z_full[i, :n_inj] = z_inj
        Z_full[i, n_inj:] = flows


    Z_full_attack = Z_full.copy()
    n_snapshots = Z_full.shape[0]

    # Define full list of target buses and accessible PG/PL meters
    target_buses_full = buses      # Buses 80–110 (31 buses)
    accessible_buses = buses      # Attacker can manipulate PG/PL meters at buses 70–118

    # PG/PL indices attacker can access (in z)
    I_access = [i for i, col in enumerate(pgpl_columns) if extract_bus_number(col) in accessible_buses]
    min_mag, max_mag = attack_mag_range
    min_meters, max_meters = meter_range
    for i in range(n_snapshots):
    # Randomly choose subset of 8–10 target buses for this row

#        k = np.random.randint(min_meters, max_meters)
        k = np.random.randint(min_meters, max_meters + 1)

        selected_target_buses = np.random.choice(target_buses_full, k, replace=False)

    # Random shift value for this row
#        shift_value = np.random.randint(min_mag, max_mag)
        shift_value = np.random.uniform(min_mag, max_mag)
        print(f"\n[Snapshot {i}]")
        print(f"  Selected buses: {sorted(selected_target_buses.tolist())}")
        print(f"  Attack magnitude (shift): {shift_value:.5f}")

    # Construct sparse c vector (117 state variables)
        c = np.zeros(H.shape[1])
        for bus in selected_target_buses:
            if bus in bus_to_state_idx:
                c[bus_to_state_idx[bus]] = shift_value

    # Compute attack vector: a = Hc
        a = H @ c

    # Mask a: zero out values attacker can't access
        for idx in range(len(a)):
            if idx not in I_access:
                a[idx] = 0.0

    # Inject into attack matrix
        Z_full_attack[i] += a

    H_inj = H[:n_buses, :]
    H_inj_pinv = pinv(H_inj)

    residuals_clean = []
    for i in range(n_rows):
        z = Z_full[i]
        z_inj = z[:n_buses]
        theta_hat = H_inj_pinv @ z_inj
        z_hat = H @ theta_hat
        residual = np.linalg.norm(z - z_hat)
        residuals_clean.append(residual)

    residuals_clean = np.array(residuals_clean)
    threshold = np.percentile(residuals_clean, 95)

    residuals_attack = []
    detected_count = 0

    for i in range(n_rows):
        z = Z_full_attack[i]
        z_inj = z[:n_buses]
        theta_hat = H_inj_pinv @ z_inj
        z_hat = H @ theta_hat
        residual = np.linalg.norm(z - z_hat)
        residuals_attack.append(residual)
        if residual > threshold:
            detected_count += 1

    stealth_count = n_rows - detected_count

    # df_attack = df.copy()

    # for i in range(n_rows):
    #     for j in range(n_buses):
    #         col = pgpl_columns[j]
    #         if col in df_attack.columns:
    #             df_attack.at[i, col] = Z_full_attack[i, j]

    df_attack = df.copy()

    for i in range(n_rows):
        for j, col in enumerate(pgpl_columns):
            if col not in df_attack.columns:
                continue

        # 🔑 CRITICAL FIX:
        # Only overwrite values that originally existed
            if not pd.isna(df_attack.at[i, col]):
                df_attack.at[i, col] = Z_full_attack[i, j]


    df_attack['Label'] = 1
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if not authentic_name.endswith(".csv"):
        authentic_name += ".csv"

    if not fdia_name.endswith(".csv"):
        fdia_name += ".csv"

    authentic_path = os.path.join(base_dir, authentic_name)
    fdia_path = os.path.join(base_dir, fdia_name)
    df.to_csv(authentic_path, index=False)
    df_attack.to_csv(fdia_path, index=False)

    return stealth_count
