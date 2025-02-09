FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3 \
    nvidia-cuda-toolkit   
    
COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

RUN export PYTHONPATH=..

# CMD ["python3", "main.py" ]