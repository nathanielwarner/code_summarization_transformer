FROM python:3.10-bullseye
ENV PYTHONUNBUFFERED 1
COPY . .
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread --log-file=- --bind=0.0.0.0:$PORT server:code_summarization_server
