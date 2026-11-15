FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    APP_MODE=ui \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install CA certificates and curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Create non-root runtime user for Cloud Run security best practices
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# Copy application source
COPY ehr_chart_summary_agent /app/ehr_chart_summary_agent
COPY physician_assist_app.py /app/physician_assist_app.py
COPY run_demo.py /app/run_demo.py
COPY README.md /app/README.md

# Ensure writable swimlane output directory owned by non-root user
RUN mkdir -p /app/outputs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/_stcore/health" || exit 1

# Supports APP_MODE=ui (default Emergency Physician Assist Streamlit UI) or APP_MODE=adk_web
CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"adk_web\" ]; then exec adk web --host 0.0.0.0 --port ${PORT:-8080}; else exec streamlit run physician_assist_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false; fi"]
