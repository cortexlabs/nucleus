FROM tensorflow/serving:2.3.0
ENV CORTEX_MODEL_SERVER_VERSION=0.1.0

RUN apt-get update -qq && apt-get install -y -q \
    curl \
    git \
    && apt-get clean -qq && rm -rf /var/lib/apt/lists/*

