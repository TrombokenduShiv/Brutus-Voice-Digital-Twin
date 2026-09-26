FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY voice_twin ./voice_twin
COPY training ./training
COPY evaluation ./evaluation
COPY models ./models
RUN pip install --no-cache-dir .
EXPOSE 8787
CMD ["uvicorn","voice_twin.api.server:app","--host","0.0.0.0","--port","8787"]
