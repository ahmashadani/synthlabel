# Create a real Dockerfile in your root folder
echo 'FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]' > Dockerfile

# Add and commit the Dockerfile
git add Dockerfile
git commit -m "Add real Dockerfile for Render deployment"
git push
