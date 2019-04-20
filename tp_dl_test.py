import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import datetime

def test_error(user, architecture):
    info = test_cache[user][architecture]
    model = models[user][architecture]['model']
    x_test, y_test, model = info['x_test'], info['y_test'], model
    y_pred = model.predict(x_test)
    if metric=='mse':
        error = mean_squared_error(y_test, y_pred)
    elif metric=='mae':
        error = mean_absolute_error(y_test, y_pred)
    return error

def train_error(user, architecture):
    info = train_cache[user][architecture]
    model = models[user][architecture]['model']
    x_train, y_train, model = info['x_train'], info['y_train'], model
    y_pred = model.predict(x_train)
    if metric=='mse':
        error = mean_squared_error(y_train, y_pred)
    elif metric=='mae':
        error = mean_absolute_error(y_train, y_pred)
    return error

def test_all():
    print('')
    print('*' * 16)
    f = open("txt/results_{0}.txt".format(datetime.datetime.now().second), "w+")
    for i in users:
        for j in range(number_of_architectures):
            e_train = round(train_error(i, j + 1), 3)
            e_test = round(test_error(i, j + 1), 3)
            print('Usuario {0} - Arquitectura {1}'.format(i, j + 1),
                  '- {0}'.format(metric), '(train): ', e_train, '- {0}(test)'.format(metric), e_test)
            f.write('Usuario {0} - Arquitectura {1}'.format(i, j + 1))
            f.write('- {0}(train): '.format(metric))
            f.write(str(e_train))
            f.write('- {0}(test)'.format(metric))
            f.write(str(e_test))
            f.write("\n")
        print('')
        f.write("\n")
    print('*' * 16)
    print('')


users = [3, 2, 57]
number_of_architectures = 6
metric='mse'
test_cache = pickle.load(open('pkl/test_cache.pkl', 'rb'))
train_cache = pickle.load(open('pkl/train_cache.pkl', 'rb'))
models = pickle.load(open('pkl/models.pkl', 'rb'))

test_all()



