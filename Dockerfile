FROM python:3.6.7

RUN mkdir /code \
&&apt-get update

COPY DSD /code
RUN pip3 install --user git+https://github.com/fizyr/keras-retinanet.git \
&&pip3 install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI \
&&pip3 install -r /code/requirements.txt
WORKDIR /code

CMD ["/bin/bash", "run.sh"]

