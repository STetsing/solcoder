FROM python:3.10-slim

ADD . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE  8080

CMD python entrypoint.py