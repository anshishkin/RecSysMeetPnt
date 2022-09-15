FROM python:3.8-slim
RUN python -m pip install --upgrade pip

WORKDIR app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m nltk.downloader punkt

ENV FLASK_APP='main.py'
ENV FLASK_DEBUG=1