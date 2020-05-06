from keras.models import Sequential
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from preprocessing.model_ready import get_train_test_data

def get_model():
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

x_train, y_train, x_test, y_test = get_train_test_data('classification', user=10)


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

model = LogisticRegression(solver='liblinear', max_iter=400, class_weight='balanced')
#model = create_classifier_model(clf)
print('LOGREG\n')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('DNN\n')
modelnn = KerasClassifier(get_model, epochs=20, batch_size=32, verbose=2, validation_data=(x_test, y_test))
modelnn.fit(x_train, y_train)
y_pred = modelnn.predict(x_test)
print(classification_report(y_test, y_pred))

print(classification_report(y_test, DummyClassifier(strategy='most_frequent', random_state=7)
                            .fit(x_train, y_train).predict(x_test)))
get_model().summary()
