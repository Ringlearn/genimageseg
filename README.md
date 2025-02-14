docker build -t cityscape-inference .
docker pull <your-username>/cityscape-inference:latest
docker run -p 5000:5000 <your-username>/cityscape-inference:latest
docker run -p 5000:5000 cityscape-inference
