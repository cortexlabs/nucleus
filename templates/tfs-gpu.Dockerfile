FROM tensorflow/serving:2.3.0-gpu
ENV CORTEX_MODEL_SERVER_VERSION=0.1.0

RUN apt-get update -qq && apt-get install -y --no-install-recommends -q \
        libnvinfer6=6.0.1-1+cuda10.1 \
        libnvinfer-plugin6=6.0.1-1+cuda10.1 \
        curl \
        git \
    && apt-get clean -qq && rm -rf /var/lib/apt/lists/*
