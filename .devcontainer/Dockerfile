FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

# install packages as root
RUN apt-get update
RUN pip install --upgrade pip

# The base image already creates a non-root vscode user (UID 1000)
USER vscode
WORKDIR /workspace

RUN git config --global safe.directory /workspace