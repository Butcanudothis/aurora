FROM python:3.11
EXPOSE 8084
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8084", "app:app", "--workers=4"]
