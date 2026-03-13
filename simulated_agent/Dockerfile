FROM python:3.11-slim

WORKDIR /app
COPY . /app/simulated_agent
RUN printf '%s\n' '#!/bin/sh' 'cd /app && exec python3 -m simulated_agent.agent_cli "$@"' > /usr/local/bin/agent-cli \
    && chmod +x /usr/local/bin/agent-cli \
    && mkdir -p /work

ENV PYTHONUNBUFFERED=1
CMD ["agent-cli", "run", "--provider", "openai"]
