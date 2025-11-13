"""
Streamlit Dashboard for Edge Caching Stackelberg Algorithms
Demonstrates all 12 plots and compares all three algorithms
"""

import sys
import os
# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms import (
    generate_instance, RUN_ALG1, RUN_ALG2, RUN_ALG3,
    jains_index
)

def display_metrics_table(metrics_data, x_label, y_label):
    """Helper function to display metrics in a table"""
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
        return df
    return None

def display_plot_info(plot_config, instance_params, description):
    """Display plot generation information"""
    st.markdown("### 📊 Algorithm Execution Details")
    st.markdown(f"**Plot Configuration:** {plot_config}")
    st.markdown(f"**Instance Parameters:** {instance_params}")
    st.markdown(f"**Description:** {description}")

# Page configuration
st.set_page_config(
    page_title="Edge Caching Algorithms Comparison",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .algorithm-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alg1 { background-color: #e3f2fd; }
    .alg2 { background-color: #f3e5f5; }
    .alg3 { background-color: #fff3e0; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("Algorithm Parameters")
    default_m = st.slider("Content Items (M)", 16, 64, 32)
    default_k = st.slider("Edge Servers (K)", 8, 32, 16)
    default_n = st.slider("Users (N)", 120, 400, 240)
    
    st.markdown("---")
    st.subheader("RAG Configuration")
    use_rag = st.checkbox("Use RAG-GenAI (Algorithm 1)", value=False)
    if use_rag:
        try:
            from edge_caching_rag_alg import configure_rag  # Already in same directory

            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
            available_docs = []
            if os.path.isdir(data_dir):
                for name in os.listdir(data_dir):
                    if name.lower().endswith((".pdf", ".txt", ".md", ".json", ".csv")):
                        available_docs.append(name)
            available_docs.sort()

            st.write("Environment variable `GROQ_API_KEY` will be used automatically if set.")
            if not available_docs:
                st.info("Place document files (e.g., PDFs) inside the `data/` folder to enable RAG.")

            selected_docs = st.multiselect(
                "Select documents from data/",
                options=available_docs,
                default=available_docs[:1] if available_docs else []
            )
            extra_doc = st.text_input("Additional document path (optional)", "")
            logs_path = st.text_input("Logs Path (optional)", "")

            if st.button("Configure RAG"):
                selected_paths = [os.path.join(data_dir, doc) for doc in selected_docs]
                if extra_doc.strip():
                    selected_paths.append(extra_doc.strip())

                if not selected_paths:
                    st.warning("Please select or provide at least one document path.")
                else:
                    configure_rag(
                        doc_paths=selected_paths,
                        logs_path=logs_path if logs_path else None,
                        groq_api_key=None
                    )
                    st.success("RAG configured with documents: " + ", ".join(selected_paths))
        except ImportError:
            st.warning("RAG module not available. Using fallback methods.")

# Main content
st.markdown('<div class="main-header">📊 Edge Caching Stackelberg Algorithms Comparison</div>', unsafe_allow_html=True)

# Algorithm descriptions
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="algorithm-box alg1">', unsafe_allow_html=True)
    st.subheader("🤖 Algorithm 1: GenAI-RAG Stackelberg")
    st.write("Uses RAG and GenAI for intelligent decision-making")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="algorithm-box alg2">', unsafe_allow_html=True)
    st.subheader("🎯 Algorithm 2: Pure Stackelberg")
    st.write("Classic Stackelberg game approach")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="algorithm-box alg3">', unsafe_allow_html=True)
    st.subheader("⚡ Algorithm 3: Greedy")
    st.write("Heuristic greedy algorithm")
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different plots
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "1. Utility vs Users", "2. Hit Ratio vs Cache", "3. Latency vs Edges",
    "4. Energy vs Size", "5. Convergence vs α", "6. RMSE vs σ",
    "7. Fairness vs Boost", "8. Utility vs Price Step", "9. Hit Ratio vs Latency",
    "10. Energy vs λ", "11. Latency-Energy Pareto", "12. Profit vs Bandwidth"
])

# Plot 1: Utility vs Users
with tab1:
    st.subheader("Utility vs Users")
    if st.button("Generate Plot", key="plot1"):
        with st.spinner("Running algorithms..."):
            Ns = (120, 180, 240, 300)
            y1, y2, y3 = [], [], []
            all_metrics = []
            
            for N in Ns:
                inst = generate_instance(N=N, M=default_m, K=default_k)
                _, _, _, h1 = RUN_ALG1(inst)
                _, _, _, h2 = RUN_ALG2(inst)
                _, _, _, h3 = RUN_ALG3(inst)
                y1.append(h1['U_L'][-1])
                y2.append(h2['U_L'][-1])
                y3.append(h3['U_L'][-1])
                all_metrics.append({
                    'N': N,
                    'Alg1': {'U_L': h1['U_L'][-1], 'hit_ratio': h1['hit_ratio'][-1], 'latency': h1['mean_latency'][-1]},
                    'Alg2': {'U_L': h2['U_L'][-1], 'hit_ratio': h2['hit_ratio'][-1], 'latency': h2['mean_latency'][-1]},
                    'Alg3': {'U_L': h3['U_L'][-1], 'hit_ratio': h3['hit_ratio'][-1], 'latency': h3['mean_latency'][-1]}
                })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(Ns))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels(Ns)
            ax.set_xlabel('Users (N)', fontsize=12)
            ax.set_ylabel('Leader Utility', fontsize=12)
            ax.set_title('Utility vs Users', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics below plot
            st.markdown("### 📊 Algorithm Execution Details")
            st.markdown(f"**Plot Configuration:** X-axis: Users (N) = {Ns}, Y-axis: Leader Utility")
            st.markdown(f"**Instance Parameters:** M={default_m}, K={default_k}")
            
            # Create metrics table
            import pandas as pd
            metrics_data = []
            for m in all_metrics:
                metrics_data.append({
                    'Users (N)': m['N'],
                    'Alg-1 Utility': f"{m['Alg1']['U_L']:.4f}",
                    'Alg-2 Utility': f"{m['Alg2']['U_L']:.4f}",
                    'Alg-3 Utility': f"{m['Alg3']['U_L']:.4f}",
                    'Alg-1 Hit Ratio': f"{m['Alg1']['hit_ratio']:.4f}",
                    'Alg-2 Hit Ratio': f"{m['Alg2']['hit_ratio']:.4f}",
                    'Alg-3 Hit Ratio': f"{m['Alg3']['hit_ratio']:.4f}"
                })
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("**How the plot was generated:**")
            st.markdown("""
            1. For each user count N in [120, 180, 240, 300]:
               - Generated a random instance with M content items and K edge servers
               - Ran Algorithm 1 (GenAI-RAG Stackelberg) and recorded final leader utility
               - Ran Algorithm 2 (Pure Stackelberg) and recorded final leader utility
               - Ran Algorithm 3 (Greedy) and recorded final leader utility
            2. Plotted the three utility curves using matplotlib
            3. Each algorithm's performance shows how leader utility scales with user count
            """)

# Plot 2: Hit Ratio vs Cache Capacity
with tab2:
    st.subheader("Hit Ratio vs Cache Capacity")
    if st.button("Generate Plot", key="plot2"):
        with st.spinner("Running algorithms..."):
            caps = (10.0, 12.0, 15.0, 18.0)
            y1, y2, y3 = [], [], []
            for cap in caps:
                inst = generate_instance(cache_cap_range=(cap, cap), M=default_m, K=default_k, N=default_n)
                _, _, _, h1 = RUN_ALG1(inst)
                _, _, _, h2 = RUN_ALG2(inst)
                _, _, _, h3 = RUN_ALG3(inst)
                y1.append(h1['hit_ratio'][-1])
                y2.append(h2['hit_ratio'][-1])
                y3.append(h3['hit_ratio'][-1])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(caps))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{cap:.1f}" for cap in caps])
            ax.set_xlabel('Cache Capacity per Edge', fontsize=12)
            ax.set_ylabel('Hit Ratio', fontsize=12)
            ax.set_title('Hit Ratio vs Cache Capacity', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, cap in enumerate(caps):
                metrics_data.append({
                    'Cache Capacity': f"{cap:.1f}",
                    'Alg-1 Hit Ratio': f"{y1[i]:.4f}",
                    'Alg-2 Hit Ratio': f"{y2[i]:.4f}",
                    'Alg-3 Hit Ratio': f"{y3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Cache Capacity = {caps}, Y-axis: Hit Ratio",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each cache capacity, generated instances and ran all three algorithms. Hit ratio measures the fraction of requests served from cache."
            )
            display_metrics_table(metrics_data, "Cache Capacity", "Hit Ratio")

# Plot 3: Latency vs Edges
with tab3:
    st.subheader("Latency vs Edges")
    if st.button("Generate Plot", key="plot3"):
        with st.spinner("Running algorithms..."):
            Ks = (8, 12, 16, 20)
            y1, y2, y3 = [], [], []
            for K in Ks:
                inst = generate_instance(K=K, M=default_m, N=default_n)
                _, _, _, h1 = RUN_ALG1(inst)
                _, _, _, h2 = RUN_ALG2(inst)
                _, _, _, h3 = RUN_ALG3(inst)
                y1.append(h1['mean_latency'][-1])
                y2.append(h2['mean_latency'][-1])
                y3.append(h3['mean_latency'][-1])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(Ks))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels(Ks)
            ax.set_xlabel('Edges (K)', fontsize=12)
            ax.set_ylabel('Mean Latency', fontsize=12)
            ax.set_title('Latency vs Edges', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, K in enumerate(Ks):
                metrics_data.append({
                    'Edges (K)': K,
                    'Alg-1 Latency': f"{y1[i]:.4f}",
                    'Alg-2 Latency': f"{y2[i]:.4f}",
                    'Alg-3 Latency': f"{y3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Edges (K) = {Ks}, Y-axis: Mean Latency",
                f"M={default_m}, N={default_n}",
                "For each number of edge servers, generated instances and ran all three algorithms. Mean latency measures average user experience delay."
            )
            display_metrics_table(metrics_data, "Edges (K)", "Mean Latency")

# Plot 4: Energy vs Size Variability
with tab4:
    st.subheader("Energy vs Size Variability")
    if st.button("Generate Plot", key="plot4"):
        with st.spinner("Running algorithms..."):
            scales = (1.0, 1.25, 1.5, 1.75)
            a, b = 1.0, 2.0
            y1, y2, y3 = [], [], []
            for s in scales:
                inst = generate_instance(size_range=(0.6*s, 1.8*s), M=default_m, K=default_k, N=default_n)
                _, _, D1, _ = RUN_ALG1(inst)
                _, _, D2, _ = RUN_ALG2(inst)
                _, _, D3, _ = RUN_ALG3(inst)
                for acc, D in ((y1, D1), (y2, D2), (y3, D3)):
                    served = D.sum()
                    misses = inst['N'] - min(inst['N'], served)
                    acc.append(a*served + b*misses)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(scales))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{s:.2f}" for s in scales])
            ax.set_xlabel('Content Size Scale', fontsize=12)
            ax.set_ylabel('Energy Proxy', fontsize=12)
            ax.set_title('Energy vs Content Size Variability', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, s in enumerate(scales):
                metrics_data.append({
                    'Size Scale': f"{s:.2f}",
                    'Alg-1 Energy': f"{y1[i]:.2f}",
                    'Alg-2 Energy': f"{y2[i]:.2f}",
                    'Alg-3 Energy': f"{y3[i]:.2f}"
                })
            display_plot_info(
                f"X-axis: Content Size Scale = {scales}, Y-axis: Energy Proxy (E = {a}*served + {b}*misses)",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each size scale, generated instances with varying content sizes. Energy proxy combines served requests and cache misses."
            )
            display_metrics_table(metrics_data, "Size Scale", "Energy Proxy")

# Plot 5: Convergence vs Alpha
with tab5:
    st.subheader("Convergence vs Alpha")
    if st.button("Generate Plot", key="plot5"):
        with st.spinner("Running algorithms..."):
            alphas = (0.2, 0.4, 0.6, 0.8)
            y1, y2, y3 = [], [], []
            for a_ in alphas:
                inst = generate_instance(M=default_m, K=default_k, N=default_n)
                _, _, _, h1 = RUN_ALG1(inst, alpha=a_)
                _, _, _, h2 = RUN_ALG2(inst)
                _, _, _, h3 = RUN_ALG3(inst)
                y1.append(h1['iters'])
                y2.append(h2['iters'])
                y3.append(h3['iters'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(alphas))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{a_:.2f}" for a_ in alphas])
            ax.set_xlabel('Blend α', fontsize=12)
            ax.set_ylabel('Outer Iterations', fontsize=12)
            ax.set_title('Convergence vs α', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, a_ in enumerate(alphas):
                metrics_data.append({
                    'Alpha (α)': f"{a_:.2f}",
                    'Alg-1 Iterations': y1[i],
                    'Alg-2 Iterations': y2[i],
                    'Alg-3 Iterations': y3[i]
                })
            display_plot_info(
                f"X-axis: Blend α = {alphas}, Y-axis: Outer Iterations",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each alpha value, ran Algorithm 1 with that blending parameter. Alpha controls the mix between GenAI predictions and behavioral demand."
            )
            display_metrics_table(metrics_data, "Alpha", "Iterations")

# Plot 6: RMSE vs Sigma
with tab6:
    st.subheader("RMSE vs Sigma")
    if st.button("Generate Plot", key="plot6"):
        with st.spinner("Running algorithms..."):
            sigmas = (0.10, 0.20, 0.30, 0.40)
            r1, r2, r3 = [], [], []
            for s in sigmas:
                a = generate_instance(M=default_m, K=default_k, N=default_n)
                b = generate_instance(M=default_m, K=default_k, N=default_n)
                a['V_uf'] = a['V_uf'] + np.random.normal(0, s, size=a['V_uf'].shape)
                _, _, D1, _ = RUN_ALG1(a, sigma_max=s)
                _, _, D1b, _ = RUN_ALG1(b, sigma_max=s)
                _, _, D2, _ = RUN_ALG2(a)
                _, _, D2b, _ = RUN_ALG2(b)
                _, _, D3, _ = RUN_ALG3(a)
                _, _, D3b, _ = RUN_ALG3(b)
                r1.append(float(np.sqrt(np.mean((D1-D1b)**2))))
                r2.append(float(np.sqrt(np.mean((D2-D2b)**2))))
                r3.append(float(np.sqrt(np.mean((D3-D3b)**2))))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(sigmas))
            width = 0.25
            ax.bar(indices - width, r1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, r2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, r3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
            ax.set_xlabel('Uncertainty σ', fontsize=12)
            ax.set_ylabel('RMSE(D)', fontsize=12)
            ax.set_title('Prediction Error vs Uncertainty', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, s in enumerate(sigmas):
                metrics_data.append({
                    'Sigma (σ)': f"{s:.2f}",
                    'Alg-1 RMSE': f"{r1[i]:.4f}",
                    'Alg-2 RMSE': f"{r2[i]:.4f}",
                    'Alg-3 RMSE': f"{r3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Uncertainty σ = {sigmas}, Y-axis: RMSE(D) between two instances",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each sigma, generated two instances (one with noise), ran algorithms on both, and computed RMSE of demand matrices to measure robustness."
            )
            display_metrics_table(metrics_data, "Sigma", "RMSE")

# Plot 7: Fairness vs Event Boost
with tab7:
    st.subheader("Fairness vs Event Boost")
    if st.button("Generate Plot", key="plot7"):
        with st.spinner("Running algorithms..."):
            boosts = (0.0, 0.2, 0.4, 0.6)
            y1, y2, y3 = [], [], []
            for b in boosts:
                inst = generate_instance(M=default_m, K=default_k, N=default_n)
                x1, _, _, _ = RUN_ALG1(inst, boost_max=b)
                x2, _, _, _ = RUN_ALG2(inst)
                x3, _, _, _ = RUN_ALG3(inst)
                y1.append(jains_index(x1))
                y2.append(jains_index(x2))
                y3.append(jains_index(x3))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(boosts))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{b:.2f}" for b in boosts])
            ax.set_xlabel('Event Boost Magnitude', fontsize=12)
            ax.set_ylabel("Jain's Fairness Index", fontsize=12)
            ax.set_title('Fairness vs Event Boost', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, b in enumerate(boosts):
                metrics_data.append({
                    'Event Boost': f"{b:.2f}",
                    'Alg-1 Fairness': f"{y1[i]:.4f}",
                    'Alg-2 Fairness': f"{y2[i]:.4f}",
                    'Alg-3 Fairness': f"{y3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Event Boost Magnitude = {boosts}, Y-axis: Jain's Fairness Index",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each boost value, ran algorithms and computed Jain's fairness index on cache allocation matrix. Higher values indicate more balanced distribution."
            )
            display_metrics_table(metrics_data, "Event Boost", "Fairness Index")

# Plot 8: Utility vs Price Step
with tab8:
    st.subheader("Utility vs Price Step")
    if st.button("Generate Plot", key="plot8"):
        with st.spinner("Running algorithms..."):
            steps = (0.05, 0.1, 0.15, 0.2)
            y1, y2, y3 = [], [], []
            for st_ in steps:
                inst = generate_instance(M=default_m, K=default_k, N=default_n)
                _, _, _, h1 = RUN_ALG1(inst, price_step=st_)
                _, _, _, h2 = RUN_ALG2(inst, price_step=st_)
                _, _, _, h3 = RUN_ALG3(inst)
                y1.append(h1['U_L'][-1])
                y2.append(h2['U_L'][-1])
                y3.append(h3['U_L'][-1])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(steps))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{st_:.2f}" for st_ in steps])
            ax.set_xlabel('Price Step Size Δp', fontsize=12)
            ax.set_ylabel('Leader Utility', fontsize=12)
            ax.set_title('Utility vs Price Step', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, st_ in enumerate(steps):
                metrics_data.append({
                    'Price Step (Δp)': f"{st_:.2f}",
                    'Alg-1 Utility': f"{y1[i]:.4f}",
                    'Alg-2 Utility': f"{y2[i]:.4f}",
                    'Alg-3 Utility': f"{y3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Price Step Size = {steps}, Y-axis: Leader Utility",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each price step size, ran algorithms with that step size for price adjustments. Larger steps allow faster convergence but may overshoot optimal prices."
            )
            display_metrics_table(metrics_data, "Price Step", "Utility")

# Plot 9: Hit Ratio vs Latency Range
with tab9:
    st.subheader("Hit Ratio vs Latency Range")
    if st.button("Generate Plot", key="plot9"):
        with st.spinner("Running algorithms..."):
            uppers = (12.0, 16.0, 20.0, 24.0)
            y1, y2, y3 = [], [], []
            for up in uppers:
                inst = generate_instance(latency_range=(3.0, up), M=default_m, K=default_k, N=default_n)
                _, _, _, h1 = RUN_ALG1(inst)
                _, _, _, h2 = RUN_ALG2(inst)
                _, _, _, h3 = RUN_ALG3(inst)
                y1.append(h1['hit_ratio'][-1])
                y2.append(h2['hit_ratio'][-1])
                y3.append(h3['hit_ratio'][-1])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(uppers))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{up:.1f}" for up in uppers])
            ax.set_xlabel('Latency Disutility Upper Bound', fontsize=12)
            ax.set_ylabel('Hit Ratio', fontsize=12)
            ax.set_title('Hit Ratio vs Latency Range', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, up in enumerate(uppers):
                metrics_data.append({
                    'Latency Upper Bound': f"{up:.1f}",
                    'Alg-1 Hit Ratio': f"{y1[i]:.4f}",
                    'Alg-2 Hit Ratio': f"{y2[i]:.4f}",
                    'Alg-3 Hit Ratio': f"{y3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Latency Upper Bound = {uppers}, Y-axis: Hit Ratio",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each latency range, generated instances with varying latency bounds and measured hit ratio performance."
            )
            display_metrics_table(metrics_data, "Latency Upper Bound", "Hit Ratio")

# Plot 10: Energy vs Lambda
with tab10:
    st.subheader("Energy vs Lambda")
    if st.button("Generate Plot", key="plot10"):
        with st.spinner("Running algorithms..."):
            lambdas = (0.1, 0.3, 0.5, 0.7)
            a, b = 1.0, 2.0
            y1, y2, y3 = [], [], []
            for lam in lambdas:
                inst = generate_instance(M=default_m, K=default_k, N=default_n)
                _, _, D1, _ = RUN_ALG1(inst, lambda_risk=lam)
                _, _, D2, _ = RUN_ALG2(inst)
                _, _, D3, _ = RUN_ALG3(inst)
                for acc, D in ((y1, D1), (y2, D2), (y3, D3)):
                    served = D.sum()
                    misses = inst['N'] - min(inst['N'], served)
                    acc.append(a*served + b*misses)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(lambdas))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels([f"{lam:.2f}" for lam in lambdas])
            ax.set_xlabel('Robustness λ', fontsize=12)
            ax.set_ylabel('Energy Proxy', fontsize=12)
            ax.set_title('Energy vs Robustness', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, lam in enumerate(lambdas):
                metrics_data.append({
                    'Lambda (λ)': f"{lam:.2f}",
                    'Alg-1 Energy': f"{y1[i]:.2f}",
                    'Alg-2 Energy': f"{y2[i]:.2f}",
                    'Alg-3 Energy': f"{y3[i]:.2f}"
                })
            display_plot_info(
                f"X-axis: Robustness λ = {lambdas}, Y-axis: Energy Proxy",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each lambda value, ran Algorithm 1 with that robustness parameter. Lambda controls risk-aversion in demand forecasting (mu_rob = mu_bar - λ*sigma)."
            )
            display_metrics_table(metrics_data, "Lambda", "Energy")

# Plot 11: Latency-Energy Pareto
with tab11:
    st.subheader("Latency-Energy Pareto")
    if st.button("Generate Plot", key="plot11"):
        with st.spinner("Running algorithms..."):
            caps = (10.0, 12.0, 15.0, 18.0)
            a, b = 1.0, 2.0
            pts1, pts2, pts3 = [], [], []
            for cap in caps:
                inst = generate_instance(cache_cap_range=(cap, cap), M=default_m, K=default_k, N=default_n)
                x1, p1, D1, h1 = RUN_ALG1(inst)
                x2, p2, D2, h2 = RUN_ALG2(inst)
                x3, p3, D3, h3 = RUN_ALG3(inst)
                for bag, D, h in ((pts1, D1, h1), (pts2, D2, h2), (pts3, D3, h3)):
                    served = D.sum()
                    misses = inst['N'] - min(inst['N'], served)
                    E = a*served + b*misses
                    bag.append((h['mean_latency'][-1], E))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter([x for x, _ in pts1], [y for _, y in pts1], label='Alg-1 GenAI-RAG', marker='o', s=100)
            ax.scatter([x for x, _ in pts2], [y for _, y in pts2], label='Alg-2 Pure Stackelberg', marker='s', s=100)
            ax.scatter([x for x, _ in pts3], [y for _, y in pts3], label='Alg-3 Greedy', marker='^', s=100)
            ax.set_xlabel('Mean Latency', fontsize=12)
            ax.set_ylabel('Energy Proxy', fontsize=12)
            ax.set_title('Latency–Energy Pareto', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, cap in enumerate(caps):
                lat1, en1 = pts1[i]
                lat2, en2 = pts2[i]
                lat3, en3 = pts3[i]
                metrics_data.append({
                    'Cache Capacity': f"{cap:.1f}",
                    'Alg-1 Latency': f"{lat1:.4f}",
                    'Alg-1 Energy': f"{en1:.2f}",
                    'Alg-2 Latency': f"{lat2:.4f}",
                    'Alg-2 Energy': f"{en2:.2f}",
                    'Alg-3 Latency': f"{lat3:.4f}",
                    'Alg-3 Energy': f"{en3:.2f}"
                })
            display_plot_info(
                f"X-axis: Mean Latency, Y-axis: Energy Proxy. Points for cache capacities: {caps}",
                f"M={default_m}, K={default_k}, N={default_n}",
                "Pareto frontier showing trade-off between latency and energy. Each point represents a different cache capacity. Lower-left is better (low latency, low energy)."
            )
            display_metrics_table(metrics_data, "Cache Capacity", "Latency-Energy")

# Plot 12: Profit per Edge vs Bandwidth
with tab12:
    st.subheader("Profit per Edge vs Bandwidth")
    if st.button("Generate Plot", key="plot12"):
        with st.spinner("Running algorithms..."):
            bands = (1800, 2200, 2600, 3000)
            y1, y2, y3 = [], [], []
            for g in bands:
                inst = generate_instance(service_cap_range=(g, g), M=default_m, K=default_k, N=default_n)
                x1, p1, D1, h1 = RUN_ALG1(inst)
                x2, p2, D2, h2 = RUN_ALG2(inst)
                x3, p3, D3, h3 = RUN_ALG3(inst)
                prof1 = float((p1*D1).sum() - (inst['C_cache']*x1).sum())/inst['K']
                prof2 = float((p2*D2).sum() - (inst['C_cache']*x2).sum())/inst['K']
                prof3 = float((p3*D3).sum() - (inst['C_cache']*x3).sum())/inst['K']
                y1.append(prof1)
                y2.append(prof2)
                y3.append(prof3)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(bands))
            width = 0.25
            ax.bar(indices - width, y1, width=width, label='Alg-1 GenAI-RAG')
            ax.bar(indices, y2, width=width, label='Alg-2 Pure Stackelberg')
            ax.bar(indices + width, y3, width=width, label='Alg-3 Greedy')
            ax.set_xticks(indices)
            ax.set_xticklabels(bands)
            ax.set_xlabel('Bandwidth Γ_e (per edge)', fontsize=12)
            ax.set_ylabel('Profit per Edge', fontsize=12)
            ax.set_title('Profit per Edge vs Bandwidth', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            st.pyplot(fig)
            
            # Display metrics
            metrics_data = []
            for i, g in enumerate(bands):
                metrics_data.append({
                    'Bandwidth (Γ_e)': g,
                    'Alg-1 Profit/Edge': f"{y1[i]:.4f}",
                    'Alg-2 Profit/Edge': f"{y2[i]:.4f}",
                    'Alg-3 Profit/Edge': f"{y3[i]:.4f}"
                })
            display_plot_info(
                f"X-axis: Bandwidth per Edge = {bands}, Y-axis: Profit per Edge (Revenue - Cache Cost) / K",
                f"M={default_m}, K={default_k}, N={default_n}",
                "For each bandwidth level, computed profit per edge as (total revenue - total cache cost) / number of edges. Shows how bandwidth affects profitability."
            )
            display_metrics_table(metrics_data, "Bandwidth", "Profit per Edge")

# Footer
st.markdown("---")
st.markdown("**Edge Caching Stackelberg Algorithms Dashboard** | Compare all three algorithms across 12 different metrics")

