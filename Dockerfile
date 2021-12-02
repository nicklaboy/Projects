FROM jupyer/pyspark-notebook

ENV JUPYTER_ENABLE_LAB=yes

RUN \
  apt-get update && \
  apt-get install -y openjdk-8-jdk && \
  rm -rf /var/lib/apt/lists/*

RUN \
    apt-get update && \
    apt-get install -y python python-dev python-pip python-virtualenv && \
    rm -rf /var/lib/apt/lists/*

RUN \
    pip install --upgrade pip && \
    pip install numpy && \
    pip install pyspark

USER $NB_UID
