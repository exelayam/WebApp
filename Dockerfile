FROM python:3.7
WORKDIR /webapp
RUN pip install pandas
COPY . /webapp
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["python", "main.py"]
