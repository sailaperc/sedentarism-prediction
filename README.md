Este repositorio contiene scripts para el análisis, preprocesamiento y seleccion y generación de características del dataset StudentLife y scripts para el creación, entrenamiento y testeo de modelos de predicción de comportamiento sedentario.

- El paquete keras-tcn fue instalado con pip dentro de ambiente de Conda
- El script train.py realiza los entrenamientos de las 4 arquitecturas
- El script test.py genera un archivo de texto con los valores del mae para el testeo de cada combinacion usuario/arquitectura
- El script generate_images.py genera las imagenes que muestran las predicciones de las arquitecturas. Las imagenes son guardadas en la carpeta Imagenes