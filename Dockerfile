FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /app
COPY . /app

CMD ["echo", "placeholder -- will be replaced by real train.py"]
