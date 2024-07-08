# syntax = docker/dockerfile:1.2
ARG VARIANT
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}
ENV PIP_CACHE_DIR=/var/cache/buildkit/pip 
RUN mkdir -p $PIP_CACHE_DIR
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt apt-get update && export DEBIAN_FRONTEND=noninteractive \ 
    && apt-get install -yqq --no-install-recommends \
    docker.io \
    cmake \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    libeigen3-dev \
    ffmpeg \
    awscli \
    xdg-utils \
    # BEGIN postgresql
    # postgresql-15 \ 
    # postgresql-contrib-15 \ 
    # END postgresql
    && rm -rf /var/lib/apt/lists/*

# Install poetry and plugins
ARG POETRY_VERSION
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install poetry==${POETRY_VERSION}

# Set up AWS SAM CLI installation with automatic architecture detection
RUN --mount=type=cache,target=/tmp/download_cache \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then \
        curl -sSL https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip -o /tmp/download_cache/aws-sam-cli.zip; \
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        curl -sSL https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-arm64.zip -o /tmp/download_cache/aws-sam-cli.zip; \
    else \
        echo "Unsupported architecture: $ARCH"; \
        exit 1; \
    fi && \
    unzip /tmp/download_cache/aws-sam-cli.zip -d /tmp/sam-installation && \
    /tmp/sam-installation/install
