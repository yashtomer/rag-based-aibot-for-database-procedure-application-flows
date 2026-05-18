FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install uv natively via pip to avoid cross-architecture binary execution issues
RUN pip install uv

# Copy pyproject.toml and uv.lock first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into the container environment
RUN uv sync --frozen --no-dev

# Install provider-specific LangChain integrations used at runtime
RUN uv pip install --python /app/.venv/bin/python --no-cache-dir langchain-openai langchain-anthropic

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI application
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

