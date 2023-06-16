# Deploying ML model in CC

The ML model is deployed using Cloud Run in this project so make sure the Cloud Run API is enabled by running this command in cloud shell
```
gcloud services enable run.googleapis.com
```

Build the docker image by running the next command
```
docker build -t [IMAGE_NAME]
```

Push the container image to a container registry by running this command
```
docker tag [IMAGE_NAME] gcr.io/[PROJECT_ID]/[IMAGE_NAME]:[TAG]
```
```
docker push gcr.io/[PROJECT_ID]/[IMAGE_NAME]:[TAG]
```

Once the container image is pushed to a container registry, it is ready to be deployed to cloud run by running this command on the shell terminal
```
gcloud run deploy [SERVICE_NAME] --image [IMAGE_URL] --platform managed --region [REGION]
```
