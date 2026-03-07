FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY mcp_server/ mcp_server/
COPY search/ search/
COPY curate/ curate/
COPY generate/ generate/
COPY integrations/ integrations/
COPY bot/ bot/
COPY paper_digest/ paper_digest/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Data dir inside container (mount your own via -v)
ENV PAPER_DISTILL_DATA_DIR=/data

EXPOSE 8765

# Default: stdio transport (for Claude Code, Cursor, etc.)
# Override with: --transport http --port 8765
ENTRYPOINT ["paper-distill-mcp"]
