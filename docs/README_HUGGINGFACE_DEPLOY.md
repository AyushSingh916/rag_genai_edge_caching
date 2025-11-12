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
   Upload these files maintaining the folder structure:
   ```
   Dockerfile
   requirements.txt
   src/
     ├── app.py
     ├── algorithms.py
     ├── rag_genai_service.py
     └── edge_caching_rag_alg.py
   README.md (optional)
   ```

3. **Configure Environment Variables (Optional)**
   In Space Settings → Variables, add:
   - `GROQ_API_KEY` - Your Groq API key for GenAI features

4. **Deploy**
   Hugging Face will automatically build and deploy your Space!

## Files Structure

```
your-space/
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── src/                    # Source code
    ├── app.py              # Streamlit dashboard
    ├── algorithms.py       # Algorithm implementations
    ├── rag_genai_service.py    # RAG service (optional)
    └── edge_caching_rag_alg.py # RAG integration (optional)
```

## Dockerfile Details

The Dockerfile:
- Uses Python 3.9 slim base
- Installs dependencies from requirements.txt
- Exposes port 7860 (Hugging Face standard)
- Runs Streamlit with proper configuration

## Features

✅ All 12 comparison plots
✅ Metrics tables below each plot
✅ Algorithm execution details
✅ Plot generation explanations
✅ Interactive sidebar controls
✅ RAG-GenAI integration (optional)

## Testing Locally

Before deploying, test locally:

```bash
# Build Docker image
docker build -t edge-caching-dashboard .

# Run container
docker run -p 7860:7860 edge-caching-dashboard

# Access at http://localhost:7860
```

## Troubleshooting

**Build fails:**
- Check Dockerfile syntax
- Verify all files are uploaded
- Check requirements.txt

**App doesn't start:**
- Check Space logs
- Verify port 7860 is exposed
- Ensure src/app.py is correct

**RAG not working:**
- Set GROQ_API_KEY in environment variables
- Fallback methods work if RAG unavailable

## Notes

- Port 7860 is required for Hugging Face Spaces
- All plots show detailed metrics and execution info
- RAG features are optional with fallbacks
- Dashboard is fully interactive

