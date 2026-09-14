FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir evalplus==0.3.1 transformers==4.45.2
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/tmp/cache
WORKDIR /output
COPY evalplus_runner.py /runner.py
