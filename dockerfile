FROM  python:3.11-slim

WORKDIR /flask_app

COPY ./model /flask_app/model/

COPY ./artifacts /flask_app/artifacts/

COPY ./app /flask_app/

RUN pip install -r requirements.txt

EXPOSE 5000

ENV image_name=app.py

CMD [ "python","app.py" ]

