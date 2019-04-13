from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from utilfunction import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, BatchNormalization, Activation

df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
#df = delete_sleep_hours(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'], axis=1, inplace=True)
df = get_user_data(df,10)

def get_model():
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


X, y = get_X_y_classification(df, True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



'''
#codigo para usar oversampling
columns = X.columns
sm = SMOTE(random_state=12, ratio='all')
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_train, columns=columns)
y_train = pd.Series(y_train)

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=350)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))
'''

clf = LogisticRegression(solver='liblinear', max_iter=400, class_weight='balanced')
model = create_model(clf)
print('LOGREG\n')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('DNN\n')
clf = KerasClassifier(get_model,epochs=20, batch_size=32, verbose=2, validation_data=(X_test,y_test))
modelnn = create_model(clf)
modelnn.fit(X_train, y_train)
y_pred = modelnn.predict(X_test)
print(classification_report(y_test, y_pred))

print(classification_report(y_test, DummyClassifier(strategy='most_frequent', random_state=7)
                           .fit(X_train,y_train).predict(X_test)))
#model.summary()
plt.close()
cols = X.columns
nums = np.arange(1,33)
a = clf.coef_
plt.plot(nums,a.reshape(-1,1))
plt.xticks(nums, cols, rotation='vertical')
plt.grid(True)
plt.show()