FROM python:3.11-slim

ADD . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE  9080

# CMD python entrypoint.py
CMD python gunicorn --bind 0.0.0.0:8080 --workers 2 wsgi:app