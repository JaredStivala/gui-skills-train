FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /app

# Avoid TF/JAX bloat from default transformers pull
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV PIP_NO_CACHE_DIR=1

# Copy first the files that change least
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Then everything else
COPY . /app/

CMD ["python", "/app/train.py"]
