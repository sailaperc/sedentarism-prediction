Trabajo Final de la optativa Introducción a las Redes Neuronales y Aprendizaje Profundo

- El ambiente usado de Conda se encuentra especificado en enviroment.yml
- El paquete keras-tcn fue instalado con pip dentro de ambiente de Conda
- El script tp_dl_train.py realiza los entrenamientos de las 6 arquitecturas (el entrenamiento sin gpu dura 40 minutos aproximadamente)
- El script tp_dl_test.py genera un archivo de texto con los valores de mse para entrenamiento y testeo de cada combinacion usuario/arquitectura
- El script generate_images.py genera las imagenes que muestran las predicciones de las arquitecturas. Las imagenes son guardadas en la carpeta Imagenes
- Los pickles models.pkl, train_cache.pkl y test_cache son generados por el script tp_dl_test.py
- models.pkl contiene un diccionario con los 18 modelos asi como tambien la historia de cuando fue entrenado
- test_cache.pkl contiene un diccionario con los atributos X_test y y_test para cada uno de los 18 modelos
- train_cache.pkl contiene un diccionario con los atributos X_train y y_train para cada uno de los 18 modelos
