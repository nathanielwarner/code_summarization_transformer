FROM tensorflow/tensorflow:2.2.0
COPY . /application/
RUN pip install -r /application/requirements.txt
EXPOSE 8000
CMD ["/bin/bash", "-c", "cd /application && gunicorn --worker-class gevent --timeout 0 --bind 0.0.0.0:8000 'server:build_server(\"models/dev/java_summ_ut_4\", \"data/leclair_java\")'"]
