FROM ubuntu:16.04
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt /app//requirements.txt


# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app//

# Expose port 8080 for Streamlit
EXPOSE 8080

CMD ["streamlit", "run", "st_tele.py"]