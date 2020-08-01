
#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

from preprocessing.datasets import get_lagged_dataset
from utils.utils import get_user_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math 

dim_learning_rate = Integer(low=-6, high=-2,name='learning_rate')
#dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=2, high=9, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')
dim_dropout = Real(low=.0, high=.8, name='dropout')

dimensions = [dim_learning_rate,
              #dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
              dim_num_epochs,
              dim_dropout]
default_parameters = [-5, 4, 'relu', 4, .3]
def log_dir_name(learning_rate, #num_dense_layers,
                 num_dense_nodes, activation, num_epochs, dropout):

    # The dir-name for the TensorBoard log-dir.
    #s = "./logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_{4}_{5}/"
    s = "./logs/lr_{0:.0e}_nodes_{1}_{2}_{3}_{4}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       #num_dense_layers,
                       num_dense_nodes,
                       activation,
                       num_epochs,
                       dropout
                       )

    return log_dir


data = get_lagged_dataset('classification','ws',51,1,1,60)
data = data.values.astype('float64')
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, shuffle=False)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

input_shape = X_train.shape[1]
def create_model(learning_rate, #num_dense_layers,
                 num_dense_nodes, activation, dropout):
    model = Sequential(name = 'mlp')
    model.add(InputLayer(input_shape=(input_shape,)))
    for i in range(num_dense_nodes,1,-1):
        name = 'layer_dense_{0}'.format(2**i)
        model.add(Dense(2**i,
                        activation=activation,
                        name=name))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

path_best_model = 'best_model.h5'

best_accuracy = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, #num_dense_layers,
            num_dense_nodes, activation,num_epochs, dropout):
    print('learning rate: {0:.1e}'.format(learning_rate))
   # print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print('num_epochs: ', num_epochs)
    print()
    num_epochs = 2**num_epochs
    learning_rate = math.pow(10,learning_rate)
    
    model = create_model(learning_rate=learning_rate,
                         #num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         dropout=dropout
                         )

    log_dir = log_dir_name(learning_rate, #num_dense_layers,
                           2**num_dense_nodes, activation, num_epochs, dropout)

    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True)

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=num_epochs,
                        batch_size=128,
                        validation_data=(X_test, y_test),
                        callbacks=[callback_log],
                        verbose=0)

    accuracy = history.history['val_accuracy'][-1]

    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()
    global best_accuracy

    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy

    del model
    K.clear_session()
    return -accuracy


#fitness(x=default_parameters)

# search_result = gp_minimize(func=fitness,
#                             dimensions=dimensions,
#                             acq_func='EI',  # Expected Improvement.
#                             n_calls=40,
#                             x0=default_parameters)

# search_result.fun

# sorted(zip(search_result.func_vals, search_result.x_iters))


from experiments.Experiment import PersonalExperiment

model = create_model(1e-3,2,'relu',.3)
model.name
pe = PersonalExperiment(model,'classification','ws',51,1,1,60,False)

# %%
pe.run()


# %%
results = pe.get_results()
np.mean(results), np.std(results)


#%%