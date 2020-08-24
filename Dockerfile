FROM tensorflow/tensorflow:2.3.0
ENV PYTHONUNBUFFERED 1
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8001
CMD gunicorn --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread --log-file=- --bind=0.0.0.0:8001 server:code_summarization_server
