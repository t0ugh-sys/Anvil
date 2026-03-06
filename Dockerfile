# LoopAgent Dockerfile
#
# Build: docker build -t loopagent .
# Run:   docker run -it --rm loopagent --goal "your goal"
#
# Supports Python 3.10+

FROM python:3.10-slim

LABEL maintainer="LoopAgent"
LABEL description="Enterprise-grade loop agent skeleton"

# Install Node.js for npm wrapper
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python source
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./
COPY requirements.txt ./

# Install Python package
RUN pip install --no-cache-dir -e .

# Copy npm wrapper
COPY bin/ ./bin/

# Make npm wrapper executable
RUN chmod +x bin/loopagent.js

# Default command
ENTRYPOINT ["python", "-m", "loop_agent.agent_cli"]
CMD ["--help"]
