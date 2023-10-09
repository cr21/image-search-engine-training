# Search Engine Model Trainer
I set up paperspace GPU instance as runner to get high processing gpu as runner. ModelTrainer Endpoint is continuous Training pipeline.
Training endpoint is expensive we can't keep it live all the time, so we will put instance in off state after training, we can trigger  workflow to put system on.

# Data Collection Architecture
![Imgur](https://i.imgur.com/wia4HB0.png)
![Imgur](https://i.imgur.com/iZOr5Eh.png)
# search Engine Training Architecture
![Imgur](https://i.imgur.com/gVjhHF1.png)

# Infrastructure Needed

1. Gpu Access on paperSpace
2. Aws S3 bucket for model Registry and Data

# Python error in paperspace
fatal error: Python.h: No such file or directory

```bash

sudo apt install libpython3.8-dev

```

# Project Setup

## Runner Setup
1. Update and upgrade the machine
2. Install the paperspace cli
3. Register Gpu as a runner
4. Add secrets
5. Done

## Actions Workflow 
1. On push checkout the code and create docker container on git-hub server.
2. Push the image to paperspace server for training
3. Once action push is completed pull and run the image on Ec2 instance.
![Imgur](https://i.imgur.com/aaYMUGo.png)




## Cost Involved

# Aws S3
    - s3 Storage: $0.025 per GB / First 50 TB / Month
    - s3 PUT : $0.005 (per 1,000 requests)
    - S3 GET : $0.0004 (per 1,000 requests)

# PaperSpace
    Gpu Machine: 
        - Ram : 30 GB
        - Cpu's: 8
        - Storage: 50 Gb
        - Gpu: 8 GB 
        - $0.462/ hour
    - For s3 bucket    :  Since we are using S3 Standard `$0.023 per GB`
    - For Ec2 Instance :  Since we are using t2.small with 20Gb storage 1vCpu and 2Gb ram `$0.0248 USD per hour`
    - For Mysql : Since we are using `$db.t3.micro` Free tier.
    - For ECR : Storage is $0.10 per GB / month for data stored in private or public repositories.
