FROM python:3.11-slim

WORKDIR /app
RUN mkdir -p /app/integrations/sandboxes/simulated \
    && touch /app/integrations/__init__.py /app/integrations/sandboxes/__init__.py
COPY __init__.py /app/integrations/sandboxes/simulated/__init__.py
COPY agent_cli.py /app/integrations/sandboxes/simulated/agent_cli.py
COPY tool_catalog.py /app/integrations/sandboxes/simulated/tool_catalog.py
RUN printf '%s\n' '#!/bin/sh' 'cd /app && exec python3 -m integrations.sandboxes.simulated.agent_cli "$@"' > /usr/local/bin/agent-cli \
    && chmod +x /usr/local/bin/agent-cli \
    && mkdir -p /work

ENV PYTHONUNBUFFERED=1
CMD ["agent-cli", "run", "--provider", "openai"]
