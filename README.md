---
title: Edge Caching Stackelberg Algorithms
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "4.0.0"
app_file: src/app.py
pinned: false
---

# Edge Caching Stackelberg Algorithms - Streamlit Dashboard

A comprehensive Streamlit dashboard for comparing three edge caching algorithms using Stackelberg game theory, ready for deployment on Hugging Face Spaces.

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

### Deploy to Hugging Face Spaces
1. Create a new Space with Docker SDK
2. Upload all files (see Deployment Checklist)
3. Set `GROQ_API_KEY` environment variable (optional)
4. Deploy!

## 📊 Features

- **12 Comparison Plots** - Visualize algorithm performance across different metrics
- **Detailed Metrics** - See exact values below each plot
- **Algorithm Execution Details** - Understand how each algorithm ran
- **Plot Generation Info** - Learn how plots were created
- **Interactive Controls** - Adjust parameters via sidebar
- **RAG-GenAI Integration** - Optional AI-powered features

## 🔬 Algorithms

1. **GenAI-RAG Stackelberg** - Uses RAG and GenAI for intelligent decision-making
2. **Pure Stackelberg** - Classic Stackelberg game approach
3. **Greedy** - Heuristic greedy algorithm

## 📈 All 12 Plots

Each plot shows:
- Visual comparison of all 3 algorithms
- Metrics table with all computed values
- Algorithm execution details
- Explanation of how the plot was generated

1. Utility vs Users
2. Hit Ratio vs Cache Capacity
3. Latency vs Edges
4. Energy vs Size Variability
5. Convergence vs Alpha
6. RMSE vs Sigma
7. Fairness vs Event Boost
8. Utility vs Price Step
9. Hit Ratio vs Latency Range
10. Energy vs Lambda
11. Latency-Energy Pareto
12. Profit per Edge vs Bandwidth

## 📁 Project Structure

```
SDN_GenAi/
├── src/                            # Source code
│   ├── app.py                      # Streamlit dashboard (main)
│   ├── algorithms.py               # Algorithm implementations
│   ├── rag_genai_service.py        # RAG service (optional)
│   └── edge_caching_rag_alg.py     # RAG integration (optional)
├── notebooks/                      # Jupyter notebooks
│   └── EdgeCaching_AdvancedPlots_AllThree (1).ipynb
├── docs/                           # Documentation
│   └── README_HUGGINGFACE_DEPLOY.md
├── Dockerfile                      # Docker config for Hugging Face
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🐳 Docker Deployment

The project includes a Dockerfile configured for Hugging Face Spaces:
- Port 7860 (Hugging Face standard)
- All dependencies installed
- Streamlit configured for headless mode

## 📝 Documentation

- `docs/README_HUGGINGFACE_DEPLOY.md` - Detailed deployment guide

## 🎯 Usage

1. Open the dashboard
2. Adjust parameters in sidebar (M, K, N)
3. Navigate to any plot tab
4. Click "Generate Plot" to see:
   - Visual comparison
   - Metrics table
   - Execution details
   - Plot generation explanation

## 🔧 Configuration

### RAG-GenAI (Optional)
- Add your PDFs/CSVs/JSONs to the `data/` folder (sample: `edge_caching_brief.pdf`)
- Default scenario dataset `edge_caching_case.json` (India vs Pakistan cricket final) is auto-loaded for reproducible results
- Check "Use RAG-GenAI" in the sidebar and select documents from the dropdown
- Set the `GROQ_API_KEY` environment variable if you want GenAI forecasting
- Click "Configure RAG" (fallback synthetic signals used if no docs/API key)

## 📊 Metrics Display

Each plot shows:
- **Plot Configuration**: X-axis and Y-axis parameters
- **Instance Parameters**: M, K, N values used
- **Metrics Table**: All computed values for each algorithm
- **Description**: How the plot was generated

## 🚢 Deployment

See `docs/README_HUGGINGFACE_DEPLOY.md` for complete deployment instructions.

## 📚 Additional Resources

- `notebooks/EdgeCaching_AdvancedPlots_AllThree (1).ipynb` - Original notebook with all algorithms and plots
