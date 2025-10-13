FROM python:3.13.8-slim

WORKDIR /app

COPY requirements.txt hf_token.json ./

RUN pip install --upgrade pip \
    && pip install uv \
    && uv pip install --system -r requirements.txt \
    && rm requirements.txt
    
# CMD ["python", "main.py"]