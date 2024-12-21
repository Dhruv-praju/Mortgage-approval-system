# Mortgage Approval Prediction System

![alt text](Img_1.png)
\
\
\
![alt text](Img_2.png)

## ML pipeline

![alt text](image.png)

## Files Overview

**2020_lar.txt** - dataset file (~10GB and ~25M rows)

**Final-GCP.py** - contains Pyspark code to build ML pipeline that contains stage like Ingestion, cleaning, Preprocessing, Feature Engineering, Model training and Model evaluation which was run on GCP cluster

**models** - folder contains trained tranformer & ML models (trained on GCP cluster)

**app.py** - contains Web UI to make predicts from input fields

## Download and run

To run the application run the following commands
```
streamlit run app.py
```