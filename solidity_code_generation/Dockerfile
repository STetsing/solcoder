FROM python:3.10-slim

ADD . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE  9080

# CMD python entrypoint.py
CMD gunicorn --workers=1 --bind 0.0.0.0:8080 wsgi:app --timeout 180