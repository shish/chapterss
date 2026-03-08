FROM python:3.14-slim

# ffmpeg for audio processing
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY chapterss/ ./chapterss/

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Expose FastAPI port
EXPOSE 8000

# Default command - use uv to run the application
CMD ["uv", "run", "--frozen", "--no-dev", "server", "--host", "0.0.0.0", "--port", "8000", "--verbose"]
