#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:07:11 2026

@author: matteo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration for a wider layout
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# --- 1. Data Loading and Preprocessing ---

@st.cache_data
def load_data(file_path):
    """
    Loads the simulation output and preprocesses it into separate dataframes.
    """
    df = pd.read_csv(file_path)
    
    # Rename the first column for clarity
    df.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
    
    # Identify Lambda columns (exclude metadata columns)
    # We assume columns that are not Metadata are the Lambda steps
    metadata_cols = ['Item', 'Prior L2 0.0', 'Type']
    lambda_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Separate data based on Type
    df_posterior = df[df['Type'] == 'Posterior'].copy().reset_index(drop=True)
    df_delta = df[df['Type'] == 'Delta'].copy().reset_index(drop=True)
    df_metrics = df[df['Type'] == 'Metrics'].copy().reset_index(drop=True)
    
    return df, df_posterior, df_delta, df_metrics, lambda_cols

# Load the data (assuming the file is in the same directory)
try:
    FILE_PATH = 'simulation_output.csv'
    full_df, df_posterior, df_delta, df_metrics, lambda_cols = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"File not found: {FILE_PATH}. Please ensure the CSV file is in the same directory.")
    st.stop()

# --- 2. Asset Grouping Definitions ---

# Define the asset lists based on the user's specification
equity_assets = [
    'MSCI Asia ex Japan', 'MSCI CH', 'MSCI EMEA', 'MSCI Europe Ex UK Ex CH',
    'MSCI Japan', 'MSCI Latam', 'MSCI North America', 'MSCI UK', 
    'Small Cap CH', 'Small Cap US'
]

bond_assets = [
    'CHF Corporate', 'CHF Sovereign', 'CHF Sovereign 10+', 
    'EM HC Corp', 'EM HC Sovi', 'EM LC Corp', 'EU High Yield', 
    'EUR Corporate', 'EUR Corporate 10+', 'EUR Sovereign', 'EUR Sovereign 10+', 
    'GBP Corporate', 'GBP Corporate 10+', 'GBP Sovereign', 'GBP Sovereign 10+', 
    'Global MBS', 'Hybrid', 'JPY Corporate', 'JPY Corporate 10+', 
    'JPY Sovereign', 'JPY Sovereign 10+', 'US High Yield', 
    'USD Corporate', 'USD Corporate 10+', 'USD Sovereign', 'USD Sovereign 10+'
]

alt_assets = [
    'Agriculture', 'Energy', 'Gold', 'HFRICRDTXXXX', 'HFRIEDI', 'HFRIEHI', 
    'HFRIEMNI', 'HFRIMI', 'HFRIMTF', 'HFRXGL', 'ILS', 
    'Industrial Metals', 'Infrastructure', 'Properties'
]

def get_asset_group(asset_name):
    if asset_name in equity_assets:
        return 'Equity'
    elif asset_name in bond_assets:
        return 'Bond'
    elif asset_name in alt_assets:
        return 'Alternatives'
    else:
        return 'Other'

# Apply grouping to Posterior and Delta dataframes
df_posterior['Group'] = df_posterior['Item'].apply(get_asset_group)
df_delta['Group'] = df_delta['Item'].apply(get_asset_group)

# --- 3. Sidebar Controls ---

st.sidebar.header("Configuration")

# Slider to select Lambda
selected_lambda = st.sidebar.select_slider(
    "Select Lambda (Regularization)",
    options=lambda_cols,
    value=lambda_cols[0]
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the Lambda value to see how the portfolio allocation and risk metrics evolve.")

# --- 4. Main Dashboard ---

st.title("Portfolio Allocation & Analysis")

# --- Section A: Portfolio Allocation (Grouped) ---
st.subheader(f"Portfolio Allocation (Lambda: {float(selected_lambda):.4f})")

col1, col2 = st.columns(2)

# Calculation: Grouped Weights
# Ensure we index by the specific order we want
grouped_posterior = df_posterior.groupby('Group')[selected_lambda].sum().reindex(['Equity', 'Bond', 'Alternatives'])
grouped_delta = df_delta.groupby('Group')[selected_lambda].sum().reindex(['Equity', 'Bond', 'Alternatives'])

# Plot 1: Absolute Allocation (Posterior)
with col1:
    st.markdown("**Absolute Allocation (Posterior)**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=grouped_posterior.index, y=grouped_posterior.values, palette="Blues_d", ax=ax1)
    ax1.set_ylabel("Weight")
    ax1.set_xlabel("")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add labels on top of bars
    for i, v in enumerate(grouped_posterior.values):
        ax1.text(i, v, f"{v:.1%}", ha='center', va='bottom')
    
    st.pyplot(fig1)

# Plot 2: Relative Allocation (Delta vs Prior)
with col2:
    st.markdown("**Relative Allocation (Delta vs Prior)**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    # Color condition: Green for positive, Red for negative
    colors = ['g' if v >= 0 else 'r' for v in grouped_delta.values]
    sns.barplot(x=grouped_delta.index, y=grouped_delta.values, palette=colors, ax=ax2)
    ax2.set_ylabel("Delta Weight")
    ax2.set_xlabel("")
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Add labels
    for i, v in enumerate(grouped_delta.values):
        offset = 0.002 if v >= 0 else -0.005
        ax2.text(i, v + offset, f"{v:.1%}", ha='center', va='bottom' if v>=0 else 'top')

    st.pyplot(fig2)

# --- Section B: Risk Metrics ---
st.markdown("---")
st.subheader("Risk Metrics Sensitivity")

# Prepare Metrics Data
metrics_plot_data = df_metrics.set_index('Item')[lambda_cols].T
metrics_plot_data.index = metrics_plot_data.index.astype(float)
current_lambda_val = float(selected_lambda)

metric_names = metrics_plot_data.columns.tolist()
cols = st.columns(len(metric_names))

for i, metric in enumerate(metric_names):
    with cols[i]:
        st.markdown(f"**{metric}**")
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Line chart
        sns.lineplot(x=metrics_plot_data.index, y=metrics_plot_data[metric], ax=ax, linewidth=2)
        
        # Highlight dot
        current_val = metrics_plot_data.loc[current_lambda_val, metric]
        ax.scatter([current_lambda_val], [current_val], color='red', s=100, zorder=5)
        
        ax.set_xlabel("Lambda")
        ax.grid(True, linestyle=':', alpha=0.6)
        sns.despine()
        
        st.pyplot(fig)
        st.metric(label="Value", value=f"{current_val:.4f}")

# --- Section C: Aggregated Allocation Area Chart ---
st.markdown("---")
st.subheader("Aggregated Allocation Evolution (Area Chart)")

# 1. Prepare data for stackplot
# Group by Asset Class and sum for all lambda columns
area_data = df_posterior.groupby('Group')[lambda_cols].sum().T

# 2. Convert Index to Float for proper x-axis scaling
area_data.index = area_data.index.astype(float)
area_data = area_data.sort_index()

# 3. Ensure columns are in the desired logical order
# Typically: Equity (riskiest) on top or bottom. Here we stack: Alternatives -> Bond -> Equity
stack_order = ['Alternatives', 'Bond', 'Equity']
area_data = area_data[stack_order]

# 4. Plotting
fig3, ax3 = plt.subplots(figsize=(12, 5))

# Create the stackplot
pal = sns.color_palette("Set2", n_colors=3) # Use a nice distinct palette
ax3.stackplot(area_data.index, 
              area_data['Alternatives'], 
              area_data['Bond'], 
              area_data['Equity'], 
              labels=['Alternatives', 'Bond', 'Equity'],
              colors=pal,
              alpha=0.85)

# Formatting
ax3.set_xlabel("Lambda (Regularization)")
ax3.set_ylabel("Total Allocation Weight")
ax3.set_xlim(area_data.index.min(), area_data.index.max())
ax3.set_ylim(0, 1.0) # Allocation usually sums to 1 (100%)
ax3.legend(loc='lower left', frameon=True)
ax3.grid(axis='y', linestyle='--', alpha=0.3)

# Add a vertical line for the currently selected Lambda
ax3.axvline(x=current_lambda_val, color='black', linestyle='--', linewidth=2, label='Current Selection')
ax3.text(current_lambda_val, 1.02, f"Lambda: {current_lambda_val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

sns.despine(left=True, bottom=False)
st.pyplot(fig3)