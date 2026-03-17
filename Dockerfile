FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy pyproject.toml and uv.lock first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into the container environment
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit application
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
