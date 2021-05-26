FROM python:3.9.5

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

ENV FLASK_APP=judge.py

CMD python3 -m flask run --host=0.0.0.0
