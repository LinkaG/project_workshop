FROM python:latest
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y git \
    && apt install build-essential -y --no-install-recommends \
    && apt-get install -y libopencv-dev

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt && \
    rm --force --recursive /var/lib/apt/lists/* && \
    rm --force --recursive /tmp/*

COPY . .

CMD [ "python3" , "main.py"]