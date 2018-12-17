FROM tensorflow/tensorflow:1.12.0-py3

ENV DSDURL="http://dsdcoreapi:8080/bg/file?id="

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends git locales libsm6 libxrender1 libxext6 \
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && locale-gen en_US.utf8 \
  && /usr/sbin/update-locale LANG=en_US.UTF-8 \
  && mkdir -p /code \
  && apt-get autoremove -y \
  && apt-get autoclean -y \
  && rm -rf /var/lib/apt/lists/*

COPY ./DSD /code

RUN pip install --user git+https://github.com/fizyr/keras-retinanet.git \
  && pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI \
  && pip install -r /code/requirements-tfcpu.txt

WORKDIR /code
EXPOSE 8000

#CMD ["/bin/bash", "run.sh"]
CMD ["python","manage.py","runserver","0.0.0.0:8000"]
