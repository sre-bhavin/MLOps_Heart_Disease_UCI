FROM python:3.9-slim

WORKDIR /code

# Copy the specialized serving requirements
COPY ./requirements_serve.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy only the serving app and the models
COPY ./app /code/app
COPY ./models /code/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]