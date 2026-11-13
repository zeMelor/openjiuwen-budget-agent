# Харнесс кодового агента на openJiuwen. Описание в README.md.
# Запуск у организаторов: --entrypoint python hack:latest /app/run_agent.py --tasks-dir /data/tasks
FROM python:3.11.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Все версии зафиксированы в requirements.lock (pip freeze из этого же базового образа),
# поэтому ставим без резолвера. uv качает пакеты параллельно:
# с обычным pip сборка шла около 8 минут при лимите 1000 с.
COPY requirements.lock .
RUN pip install uv==0.12.24 \
 && uv pip install --system --no-cache --no-deps -r requirements.lock \
 && pip uninstall -y uv

# Контейнер запускается с --read-only: байткод компилируем заранее, при запуске его не записать.
RUN python -m compileall -q -j 0 /usr/local/lib/python3.11/site-packages > /dev/null || true

COPY run_agent.py .
COPY budget_agent ./budget_agent
RUN python -m compileall -q /app

CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
