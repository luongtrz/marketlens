FROM python:3.12-slim

WORKDIR /app

COPY requirements-demo.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY stockmem /app/stockmem
COPY demo.py /app/demo.py

ENV VECTOR_BACKEND=memory
ENV DB_URL=sqlite+aiosqlite:///test.db

EXPOSE 8000

CMD ["python", "demo.py"]
