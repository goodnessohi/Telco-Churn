#Use the official Python image from Docker
FROM python:3.9-slim

#Set the working directory to /app
WORKDIR /app

#Copy the requirements file into the container
COPY requirements.txt .

#Install the required Python packages
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

#Copy the rest of the application code to the computer
COPY . .

#Expose the port that Streamlit app runs on
EXPOSE 8501

#Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]