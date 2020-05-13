import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)

numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline', 'hourofday', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount']

X = pd.read_pickle('regressionXsamples.pkl')
y = pd.read_pickle('regressionysamples.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
ss = StandardScaler()
X_train[numeric_cols] = ss.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = ss.transform(X_test[numeric_cols])
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
plt.plot(reg.coef_)
plt.xticks(np.arange(0, len(X.columns)), X.columns, rotation='vertical')
plt.show()
y_pred = reg.predict(X_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)
plt.plot(reg.coef_)
plt.xticks(np.arange(0, len(X.columns)), X.columns, rotation='vertical')
plt.show()
y_pred = reg.predict(X_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

n_alphas = 20
alphas = np.logspace(-2, 2, n_alphas)
reg = linear_model.RidgeCV(alphas=alphas, cv=10)
reg.fit(X_train, y_train)


n_alphas = 100
alphas = np.logspace(-2, 6, n_alphas)
coefs = []
errors = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    y_pred = reg.predict(X_test)
    errors.append((mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.show()


reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

X_train = PolynomialFeatures(interaction_only=True).fit_transform(X_train)
clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,
                 shuffle=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


def baseline_model():
    model = Sequential()
    model.add(Dense(256, input_dim=27, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256, input_dim=27, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


# fix random seed for reproducibility

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=32, verbose=2)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=20, batch_size=256, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))