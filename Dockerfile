FROM python:3.12-slim

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR /app

COPY ./app.py /app/app.py

CMD ["streamlit", "run", "app.py"]