FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY python /app/python
RUN python -m pip install --no-cache-dir "/app/python[api,data]"

RUN useradd --create-home astromap
USER astromap
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "astromap.api:app", "--host", "0.0.0.0", "--port", "8001"]
