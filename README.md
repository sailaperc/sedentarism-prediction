Este repositorio contiene scripts para el análisis, preprocesamiento y seleccion y generación de características del dataset StudentLife y scripts para el creación, entrenamiento y testeo de modelos de predicción de comportamiento sedentario.

- El paquete keras-tcn fue instalado con pip dentro de ambiente de Conda
- El script train.py realiza los entrenamientos de las 4 arquitecturas
- El script test.py genera un archivo de texto con los valores del mae para el testeo de cada combinacion usuario/arquitectura
- El script generate_images.py genera las imagenes que muestran las predicciones de las arquitecturas. Las imagenes son guardadas en la carpeta Imagenes



#The preprocessing has several stages: 
* Step 1: Generate the features from the raw data for a specific time granularity (preprocessing_studentlife_raw)
* Step 2: Delete user 52, make dummy features, delete sleep hours, calculate MET level and/or MET classes (preprocessing_various)
* Step 3: Generate lagged dataset with a specific period and a specific number of lagged (preprocessing_lagged_dataset)
* Step 4: Model specific preprocessings (Regression/Classification). Split train/split, split x and y (preprocessing_model_ready)