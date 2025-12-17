#!/bin/bash

# Start FastAPI in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground (or background if we want to wait)
# Streamlit runs on port 8501 by default
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

