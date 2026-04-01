FROM python:3.12-slim

WORKDIR /app

# System deps: libgomp1 is required by LightGBM for multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy code
COPY src/       src/
COPY app/       app/
COPY env_setup.py .

# Bake in pre-trained models, figures, CSVs, and data so the image is
# fully self-contained — judges run `docker run` with no extra setup.
# Run `make train && make run-compare && make run-shap` locally first,
# then `make app-build` to package everything into the image.
COPY outputs/      outputs/
COPY data/         data/
COPY dataset_test/ dataset_test/

# Tell the app where the project root is
ENV DATABATTLE_ROOT=/app
ENV MPLBACKEND=Agg

EXPOSE 8501

# Healthcheck so docker-compose knows when the app is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
