import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
np.random.seed(7)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.linear_model import LogisticRegression
from preprocessing.studentlife_raw import student_data_preprocessing
from preprocessing.datasets import get_lagged_dataset
from personal_impersonal import per_user, live_one_out

def show_metric(title, ylabel, labels, data, markers=('s','^'), colors=('b','r')):
    df = student_data_preprocessing()
    users = np.arange(1, 49)
    userslabel = df.index.get_level_values(0).drop_duplicates()
    plt.close()
    for d,m,c in zip(data,markers,colors):
        plt.scatter(users, d, marker=m, c=c)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('User')
    plt.legend(labels,
               loc='best')
    plt.xticks(users, userslabel, rotation='vertical')
    plt.ylim(0.55,1)
    plt.grid(True)
    plt.show()

show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withoutsleep'), pd.read_pickle('f1_p_nn_withoutsleep')])


'''
df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = delete_sleep_hours(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)
'''

def baseline_model():
    estimator = Sequential([
    Dense(256,input_dim=28,kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, kernel_initializer='uniform',kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(32, kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation='sigmoid')
    ])
    estimator.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return estimator


df = get_lagged_dataset('classification')

modelnn = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=512, verbose=2)
modelnn2 = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=32, verbose=2)

modellr = LogisticRegression(solver='liblinear', max_iter=400, class_weight='balanced')

model_type = 'classification'
b = per_user(df, modelnn, 'classification')

'''


"""
show_metric('Model F1-score for impersonal models ',
            'F1-score',
            ['nn', 'logreg'],
            [f1_imp_nn, f1_imp_logreg])
"""

show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withsleep'), pd.read_pickle('f1_p_logreg_withsleep')])

show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withoutsleep'), pd.read_pickle('f1_p_logreg_withoutsleep')])
'''

