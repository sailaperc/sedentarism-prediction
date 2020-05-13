from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from numpy.random import seed
seed(7)

df = pd.read_pickle('sedentarismunshifted.pkl')
df = makeSedentaryClasses(df)

size = df.shape[1]
# Initialize the constructor

def build_nn():
    nn = Sequential([
        Dense(64, activation='relu', input_shape=(size,)),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    nn.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['categorical_accuracy'])
    #clf = KerasClassifier(nn,epochs=20, batch_size=256, verbose=2)
    return nn

'''
shifts = [1,2,3,4,24,7*24, 7*24*4]
meanprecision = []
meanrecall = []
for shift in shifts:
    print('shift nro ', shift)
    dfshifted = shift_hours(df,shift)
    i = 0
    precision = []
    recall = []
    logo = LeaveOneGroupOut()
    groups = dfshifted.index.get_level_values(0)
    X, y = get_X_y_classification(dfshifted)
    y = to_categorical(y)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0,
                  validation_data=(X_test, y_test))
        y_pred = model.predict(X_test)
        precision.append(precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted'))
        recall.append(recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted'))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    meanprecision.append(np.mean(precision))
    meanrecall.append(np.mean(recall))

shifts = np.linspace(0,24,25)
#shifts = [0, 1, 2, 3, 4, 24 ,7 *24 , 7*24*4]
meanprecision = []
meanrecall = []
clf = KerasClassifier(build_fn=build_nn, epochs=20, batch_size=256, verbose=2)
for shift in shifts:
    model = create_model(clf)
    print('shift nro ', shift)
    dfshifted = shift_hours(df,shift)
    precision = []
    recall = []
    X, y = get_X_y_classification(dfshifted)
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    print(y_pred)
    precision.append(precision_score(np.argmax(y_test, axis=1), y_pred, average='weighted'))
    recall.append(recall_score(np.argmax(y_test, axis=1), y_pred, average='weighted'))
    print(classification_report(np.argmax(y_test, axis=1), y_pred))
    meanprecision.append(np.mean(precision))
    meanrecall.append(np.mean(recall))
'''



shifts = np.linspace(0,24,25)
#shifts = [0, 1, 2, 3, 4, 24 ,7 *24 , 7*24*4]
precision = []
recall = []
for shift in shifts:
    clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)
    model = create_model(clf)
    dfshifted = shift_hours(df,shift, 'classification')
    X, y = get_X_y_classification(dfshifted,False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    print(y_pred)
    precision.append(precision_score(y_test, y_pred, average='weighted'))
    recall.append(recall_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))



precision2 = []
recall2 = []
for shift in shifts:
    clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)
    model = create_model(clf)
    print('shift nro ', shift)
    dfshifted = shift_hours(df,shift)
    X, y = get_X_y_classification(dfshifted,True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    print(y_pred)
    precision2.append(precision_score(y_test, y_pred, average='weighted'))
    recall2.append(recall_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))




plt.close()
N = len(shifts)
x = np.arange(N)
plt.scatter(x,precision,label='precision')
plt.scatter(x,precision2,label='precision2')
plt.scatter(x,recall,label='recall')
plt.scatter(x,recall2,label='recall2')
plt.xticks(x, shifts, rotation='vertical')
plt.legend()
plt.show()


