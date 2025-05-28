# skin-cancer-classification

## Table of Contents
- [Local Setup](#local-setup)
  - [Training the Model](#training-the-model)
- [Docker Setup](#docker-setup)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running the Docker Container](#running-the-docker-container)
- [Using DockerHub Image](#using-dockerhub-image)
  - [Pulling the Image](#pulling-the-image)
  - [Running the Container](#running-the-container)

## Local Setup

Clone the repository:

```bash
git clone https://github.com/Enamul16012001/skin-cancer-classification.git
cd skin-cancer-classification
```

Create a virtual environment:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, run the training script:

#### On Windows:
```bash
python train.py
```

#### On Ubuntu/Linux:
```bash
python3 train.py
```

If you encounter any issues on Ubuntu, ensure you have proper permissions:
```bash
chmod +x train.py
python3 train.py
```


## Docker Setup

### Building the Docker Image

Build the Docker image using the Dockerfile provided in the repository:

```bash
docker build -t skin-cancer-classification .
```

### Running the Docker Container

Run the Docker container with:

```bash
docker run -p 8000:8000 skin-cancer-classification
```

This will:
1. Start the container
2. Expose the prediction API on port 8000
3. Load the pre-trained model

## Using DockerHub Image

### Pulling the Image

Pull the pre-built Docker image from DockerHub:

```bash
docker pull enamulatiq/skin-cancer-classification:latest
```

### Running the Container

Run the container from the DockerHub image:

```bash
docker run -p 8000:8000 enamulatiq/skin-cancer-classification
```