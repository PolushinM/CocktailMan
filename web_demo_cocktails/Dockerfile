FROM python:3.9-slim as builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.9-slim
RUN addgroup --gid 1001 --system app && \
    adduser --shell /bin/false --disabled-password --uid 1001 --system --group app
WORKDIR /home/app
ENV PATH="/home/app/.local/bin:$PATH"
COPY --from=builder /app/wheels /wheels
RUN pip install --upgrade pip
COPY . /home/app
RUN chown -R app: /home/app
USER app
RUN pip install --user --no-cache /wheels/*
CMD ["gunicorn", "init_docker:app"]
