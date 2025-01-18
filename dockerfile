FROM ubuntu:24.04
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    pip3 \
    python3 \
    tensorflow-gpu \    
