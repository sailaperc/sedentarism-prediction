import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()
np.random.seed(7)
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.linear_model import LogisticRegression
from preprocessing.studentlife_raw import student_data_preprocessing
from preprocessing.datasets import get_lagged_dataset
from preprocessing.various import delete_sleep_hours
from personal_impersonal import per_user, live_one_out
import pickle as pkl
import time
from utils import file_exists


def show_metric(title, ylabel, labels, data, markers=('s', '^'), colors=('b', 'r')):
    df = student_data_preprocessing()
    users = np.arange(1, 49)
    userslabel = df.index.get_level_values(0).drop_duplicates()
    plt.close()
    for d, m, c in zip(data, markers, colors):
        plt.scatter(users, d, marker=m, c=c)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('User')
    plt.legend(labels,
               loc='best')
    plt.xticks(users, userslabel, rotation='vertical')
    plt.ylim(0.55, 1)
    plt.grid(True)
    plt.show()


'''
show_metric('',
            'F1-Score',
            ['Impersonal', 'Personal'],
            [pd.read_pickle('f1_imp_nn_withoutsleep'), pd.read_pickle('f1_p_nn_withoutsleep')])
'''


def baseline_model():
    estimator = Sequential([
        Dense(256, input_dim=28, kernel_initializer='uniform', kernel_regularizer='l2', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dense(128, kernel_initializer='uniform', kernel_regularizer='l2', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dense(64, kernel_initializer='uniform', kernel_regularizer='l2', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dense(32, kernel_initializer='uniform', kernel_regularizer='l2', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='sigmoid')
    ])
    estimator.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['binary_accuracy'])
    return estimator


def run_classification_models():
    times = {}
    for kdata in datasets.keys():
        data = datasets[kdata]
        for t in ["personal", "impersonal"]:
            for kmodel in models.keys():
                run_info = f'{model_type}_{kdata}_{t}_{kmodel}'
                filename = f'pkl/results/{run_info}.pkl'
                if (file_exists(filename)):
                    print(f'{run_info} models already tested... skipping')
                else:
                    print(f'Running {model_type} for model {t} and {kmodel}, with data {kdata}')
                    start = time.time()

                    model = models[kmodel][t]

                    if t == 'personal':
                        results = per_user(data, model, model_type)
                    else:
                        results = live_one_out(data, model, model_type)

                    end = time.time()
                    total = round((end - start) / 60, 3)
                    times[f'{model_type}_{kdata}_{t}_{kmodel}'] = total
                    print(f'Run in {total} minutes')
                    print('*\n' * 3)

                    results_file = open(filename, 'wb')
                    pkl.dump(results, results_file)
                    results_file.close()

                    tracker.print_diff()
    return times


def show_classification_results():
    for sh in ["ws", "wos"]:
        for t in ["personal", "impersonal"]:
            for k in models.keys():
                print(pkl.load(open(f'pkl/results/{model_type}_{sh}_{t}_{k}.pkl', 'rb')))


model_type = 'classification'
df_ws = get_lagged_dataset(model_type)
df_wos = delete_sleep_hours(df_ws)

datasets = {'ws': df_ws, 'wof': df_wos}

modelnnImp = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=512, verbose=0)
modelnnPer = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=32, verbose=0)
modellr = LogisticRegression(solver='liblinear', max_iter=400, class_weight='balanced')

models = {"lr": {"personal": modellr, "impersonal": modellr},
          "nn": {"personal": modelnnPer, "impersonal": modelnnImp}}

if __name__ == '__main__':
    run_classification_models()
