FROM python:3

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

#predownload models
COPY download_models.py .
RUN python download_models.py

COPY . .

EXPOSE 80

CMD [ "python", "-u", "main.py" ]
