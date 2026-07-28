# Anvil Dockerfile
#
# Build: docker build -t anvil .
# Run:   docker run -it --rm anvil --goal "your goal"
#
# Supports Python 3.11+

FROM python:3.11-slim

LABEL maintainer="Anvil"
LABEL description="Terminal-first coding agent runtime"

# Install Node.js for npm wrapper
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python source and dependencies
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./
COPY requirements.txt ./
COPY skills/ ./skills/

# Install Python package (non-editable for production)
RUN pip install --no-cache-dir .

# Copy npm wrapper
COPY bin/ ./bin/

# Make npm wrapper executable
RUN chmod +x bin/anvil.js

# Default command
ENTRYPOINT ["python", "-m", "anvil.entrypoints.agent"]
CMD ["--help"]
