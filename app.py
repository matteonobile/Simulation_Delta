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

# Set page configuration
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# --- 1. Data Loading and Preprocessing ---

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
    
    metadata_cols = ['Item', 'Prior L2 0.0', 'Type']
    lambda_cols = [c for c in df.columns if c not in metadata_cols]
    
    df_posterior = df[df['Type'] == 'Posterior'].copy().reset_index(drop=True)
    df_delta = df[df['Type'] == 'Delta'].copy().reset_index(drop=True)
    df_metrics = df[df['Type'] == 'Metrics'].copy().reset_index(drop=True)
    
    return df, df_posterior, df_delta, df_metrics, lambda_cols

try:
    FILE_PATH = 'simulation_output.csv'
    full_df, df_posterior, df_delta, df_metrics, lambda_cols = load_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"File not found: {FILE_PATH}. Please ensure the CSV file is in the same directory.")
    st.stop()

# --- 2. Asset Grouping ---

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
    if asset_name in equity_assets: return 'Equity'
    elif asset_name in bond_assets: return 'Bond'
    elif asset_name in alt_assets: return 'Alternatives'
    else: return 'Other'

df_posterior['Group'] = df_posterior['Item'].apply(get_asset_group)
df_delta['Group'] = df_delta['Item'].apply(get_asset_group)

# --- 3. Sidebar ---

st.sidebar.header("Configuration")
selected_lambda = st.sidebar.select_slider(
    "Select Lambda (Regularization)",
    options=lambda_cols,
    value=lambda_cols[0]
)
st.sidebar.markdown("---")
st.sidebar.info("Adjust Lambda to see portfolio evolution.")

# --- 4. Main Dashboard ---

st.title("Portfolio Allocation & Analysis")

# Create Tabs
tab1, tab2 = st.tabs(["Overview (Aggregated)", "Detailed Allocation"])

# ==========================================
# TAB 1: OVERVIEW (Aggregated & Metrics)
# ==========================================
with tab1:
    st.subheader(f"Portfolio Allocation (Lambda: {float(selected_lambda):.4f})")
    
    col1, col2 = st.columns(2)
    
    # Aggregation
    grouped_posterior = df_posterior.groupby('Group')[selected_lambda].sum().reindex(['Equity', 'Bond', 'Alternatives'])
    grouped_delta = df_delta.groupby('Group')[selected_lambda].sum().reindex(['Equity', 'Bond', 'Alternatives'])
    
    # Plot 1: Aggregated Posterior
    with col1:
        st.markdown("**Absolute Allocation (Posterior)**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        # Fix for palette warning: set hue=x and legend=False
        sns.barplot(x=grouped_posterior.index, y=grouped_posterior.values, 
                    hue=grouped_posterior.index, palette="Blues_d", ax=ax1, legend=False)
        ax1.set_ylabel("Weight")
        ax1.set_xlabel("")
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        for i, v in enumerate(grouped_posterior.values):
            ax1.text(i, v, f"{v:.1%}", ha='center', va='bottom')
        st.pyplot(fig1)

    # Plot 2: Aggregated Delta
    with col2:
        st.markdown("**Relative Allocation (Delta vs Prior)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ['g' if v >= 0 else 'r' for v in grouped_delta.values]
        # For custom colors per bar based on value, we can't easily use hue=index. 
        # Instead, we pass the color list directly to 'palette' which is still valid if hue is not used, 
        # OR we use 'hue' with a custom dictionary. 
        # Safest simple fix for boolean coloring: just use 'palette' directly as list, usually acceptable, 
        # but to be strictly compliant we can use barplot without palette and set facecolors manually, or:
        sns.barplot(x=grouped_delta.index, y=grouped_delta.values, 
                    hue=grouped_delta.index, palette=colors, ax=ax2, legend=False)
        
        ax2.set_ylabel("Delta Weight")
        ax2.set_xlabel("")
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        for i, v in enumerate(grouped_delta.values):
            offset = 0.002 if v >= 0 else -0.005
            ax2.text(i, v + offset, f"{v:.1%}", ha='center', va='bottom' if v>=0 else 'top')
        st.pyplot(fig2)

    # Risk Metrics
    st.markdown("---")
    st.subheader("Risk Metrics Sensitivity")
    metrics_plot_data = df_metrics.set_index('Item')[lambda_cols].T
    metrics_plot_data.index = metrics_plot_data.index.astype(float)
    current_lambda_val = float(selected_lambda)
    
    metric_names = metrics_plot_data.columns.tolist()
    m_cols = st.columns(len(metric_names))
    
    for i, metric in enumerate(metric_names):
        with m_cols[i]:
            st.markdown(f"**{metric}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.lineplot(x=metrics_plot_data.index, y=metrics_plot_data[metric], ax=ax, linewidth=2)
            current_val = metrics_plot_data.loc[current_lambda_val, metric]
            ax.scatter([current_lambda_val], [current_val], color='red', s=100, zorder=5)
            ax.set_xlabel("Lambda")
            ax.grid(True, linestyle=':', alpha=0.6)
            sns.despine()
            st.pyplot(fig)
            st.metric(label="Value", value=f"{current_val:.4f}")

    # Area Chart
    st.markdown("---")
    st.subheader("Aggregated Allocation Evolution")
    area_data = df_posterior.groupby('Group')[lambda_cols].sum().T
    area_data.index = area_data.index.astype(float)
    area_data = area_data.sort_index()
    area_data = area_data[['Alternatives', 'Bond', 'Equity']]
    
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    pal = sns.color_palette("Set2", n_colors=3)
    ax3.stackplot(area_data.index, 
                  area_data['Alternatives'], 
                  area_data['Bond'], 
                  area_data['Equity'], 
                  labels=['Alternatives', 'Bond', 'Equity'],
                  colors=pal, alpha=0.85)
    ax3.set_xlabel("Lambda")
    ax3.set_ylabel("Weight")
    ax3.set_xlim(area_data.index.min(), area_data.index.max())
    ax3.set_ylim(0, 1.0)
    ax3.legend(loc='lower left', frameon=True)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.axvline(x=current_lambda_val, color='black', linestyle='--', linewidth=2)
    sns.despine(left=True)
    st.pyplot(fig3)

# ==========================================
# TAB 2: DETAILED ALLOCATION
# ==========================================
with tab2:
    st.subheader(f"Detailed Asset Allocation (Lambda: {float(selected_lambda):.4f})")
    st.markdown("Posterior weights are **normalized to 100%** within each group. Delta weights are **absolute**.")

    groups_to_show = ['Equity', 'Bond', 'Alternatives']
    
    for group in groups_to_show:
        st.markdown(f"### {group}")
        
        # Filter Data
        sub_post = df_posterior[df_posterior['Group'] == group].copy()
        sub_delta = df_delta[df_delta['Group'] == group].copy()
        
        # Sort by weight (descending) for better visual
        sub_post = sub_post.sort_values(by=selected_lambda, ascending=False)
        # Align delta order with posterior order for consistency
        sub_delta = sub_delta.set_index('Item').reindex(sub_post['Item']).reset_index()

        # Normalize Posterior to 100% within group
        total_weight = sub_post[selected_lambda].sum()
        if total_weight > 0:
            sub_post['Normalized'] = (sub_post[selected_lambda] / total_weight) * 100
        else:
            sub_post['Normalized'] = 0

        c1, c2 = st.columns(2)
        
        # Detailed Posterior Plot
        with c1:
            st.markdown(f"**Composition ({group})**")
            fig_d1, ax_d1 = plt.subplots(figsize=(10, len(sub_post)*0.4 + 2))
            
            sns.barplot(
                data=sub_post,
                y='Item',
                x='Normalized',
                hue='Item', # Fix for palette warning
                palette="Blues_r",
                ax=ax_d1,
                legend=False
            )
            ax_d1.set_xlabel("% of Group")
            ax_d1.set_ylabel("")
            ax_d1.grid(axis='x', linestyle='--', alpha=0.5)
            
            # Labels
            for i, v in enumerate(sub_post['Normalized']):
                ax_d1.text(v + 0.5, i, f"{v:.1f}%", va='center')
            
            sns.despine(left=True, bottom=False)
            st.pyplot(fig_d1)

        # Detailed Delta Plot
        with c2:
            st.markdown(f"**Active Bets ({group})**")
            fig_d2, ax_d2 = plt.subplots(figsize=(10, len(sub_delta)*0.4 + 2))
            
            # Color logic for deltas
            delta_vals = sub_delta[selected_lambda].values
            d_colors = ['g' if v >= 0 else 'r' for v in delta_vals]
            
            sns.barplot(
                data=sub_delta,
                y='Item',
                x=selected_lambda,
                hue='Item', # Fix for warning
                palette=d_colors,
                ax=ax_d2,
                legend=False
            )
            ax_d2.set_xlabel("Delta Weight (Absolute)")
            ax_d2.set_ylabel("")
            ax_d2.axvline(0, color='black', linewidth=0.8)
            ax_d2.grid(axis='x', linestyle='--', alpha=0.5)
            
            # Labels
            for i, v in enumerate(delta_vals):
                offset = 0.001 if v >= 0 else -0.001
                ha = 'left' if v >= 0 else 'right'
                ax_d2.text(v + offset, i, f"{v:.2%}", va='center', ha=ha)
                
            sns.despine(left=True, bottom=False)
            st.pyplot(fig_d2)
            
        st.markdown("---")