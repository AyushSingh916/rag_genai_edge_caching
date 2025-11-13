# Deploying to Hugging Face Spaces

## Quick Start

1. **Create a Hugging Face Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select:
     - **SDK**: Docker
     - **Hardware**: CPU Basic (or higher)
     - **Visibility**: Public or Private

2. **Upload Required Files**
   Upload these files while preserving the folder structure:
   ```
   Dockerfile
   requirements.txt
   src/
     ├── app.py
     ├── algorithms.py
     ├── rag_genai_service.py
     └── edge_caching_rag_alg.py
   data/
     ├── edge_caching_brief.pdf
     └── edge_caching_case.json
   README.md (optional)
   ```

3. **Configure Environment Variables (Optional)**
   - Set `GROQ_API_KEY` in the Space settings if you want GenAI forecasting

4. **Deploy**
   - Hugging Face will automatically build and serve the Space

## File Structure

```
your-space/
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── src/                    # Source code (Streamlit app + algorithms)
└── data/                   # RAG documents (PDF/CSV/JSON)
```

## Dockerfile Details

The Dockerfile:
- Uses Python 3.9 slim base image
- Installs dependencies from `requirements.txt`
- Copies the `src/` directory into the container
- Exposes port 7860 (Streamlit default on Hugging Face)
- Runs `streamlit run src/app.py`

## Features

✅ RAG-GenAI Stackelberg algorithm (Algorithm 1)
✅ Pure Stackelberg algorithm (Algorithm 2)
✅ Greedy baseline algorithm (Algorithm 3)
✅ 12 comparison plots with metrics tables and explanations
✅ PDF selection directly from `data/` folder
✅ Bundled scenario dataset (`edge_caching_case.json`) auto-loads for reproducible runs

## Local Testing

```bash
# Build the image
docker build -t edge-caching-dashboard .

# Run the container
docker run -p 7860:7860 edge-caching-dashboard

# Open http://localhost:7860
docker logs -f <container-id>
```

## Troubleshooting

**Build fails:**
- Check Dockerfile syntax
- Verify all files are uploaded
- Confirm folder structure matches the repository

**App doesn't start:**
- Inspect Space logs
- Confirm port 7860 is exposed
- Ensure `src/app.py` exists and imports work

**RAG not engaging:**
- Verify `data/` contains documents (e.g., `edge_caching_brief.pdf`)
- Ensure `Use RAG-GenAI` is checked and documents selected in the UI
- Set `GROQ_API_KEY` if GenAI forecasting is required

## Notes

- The Streamlit UI automatically lists PDFs from the `data/` folder
- Without `GROQ_API_KEY`, the app falls back to deterministic synthetic data
- You can add additional PDFs/CSVs/JSONs to `data/` to customize behavior
