FROM python:3.9.9-slim
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
&& pip install --no-cache-dir -r requirements.txt