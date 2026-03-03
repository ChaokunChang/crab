FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY agent_workload.py /app/agent_workload.py
ENV OUT_DIR=/work
CMD ["python", "/app/agent_workload.py"]