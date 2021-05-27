FROM python:3.9.5

WORKDIR /app

ADD requirements.txt /app

RUN pip install -r requirements.txt

ADD drop_and_create_db.sh judge.py nctu_oauth.py predictor.py schema.sql /app/

ADD templates /app/templates

ENV TZ=Asia/Taipei

CMD uwsgi --http 0.0.0.0:5000 --module judge:app
