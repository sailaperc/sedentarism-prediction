# Conda env
enviroment.yml contains all necessary python packages (uses python 3.7)
In anaconda propt run: 
* conda env create -f environment.yml
* conda activate tesis

# The preprocessing has several stages: 
* Step 1: Generate the features from the raw data for a specific time granularity (preprocessing.studentlife_raw)
* Step 2: Delete user 52, make dummy features, delete sleep hours, calculate MET level and/or MET classes (preprocessing.various)
* Step 3: Get a specific dataset based on: granularity, period, number of lags, model type, with sleep buckets and user (preprocessing.datasets)
* Step 4: Model specific preprocessings (Regression/Classification). Split train/split, split x and y (preprocessing.model_ready)

# Models
## Models.classification
Run classification_models_comparison.py to generate train and test all the models for classification

