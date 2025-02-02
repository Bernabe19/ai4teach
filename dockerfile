FROM ubuntu:24.04
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    pip3 \
    python3 \
    python-multipart \
    tensorflow   
    
COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

RUN export PYTHONPATH=..

CMD ["python3", "main.py" ]