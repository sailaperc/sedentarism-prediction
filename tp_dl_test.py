import pickle
from sklearn.metrics import mean_squared_error
import datetime



def test_error(user, architecture):
    info = test_cache[user][architecture]
    model = models[user][architecture]['model']
    x_test, y_test, model = info['x_test'], info['y_test'], model
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def train_error(user, architecture):
    info = train_cache[user][architecture]
    model = models[user][architecture]['model']
    x_train, y_train, model = info['x_train'], info['y_train'], model
    y_pred = model.predict(x_train)
    return mean_squared_error(y_train, y_pred)


def test_all():
    print('')
    print('*' * 16)
    f = open("txt/results_{0}.txt".format(datetime.datetime.now().second), "w+")
    for i in users:
        for j in range(number_of_architectures):
            mse_train = round(train_error(i, j + 1), 3)
            mse_test = round(test_error(i, j + 1), 3)
            print('Usuario {0} - Arquitectura {1}'.format(i, j + 1),
                  '- mse(train): ', mse_train, '- mse(test)', mse_test)
            f.write('Usuario {0} - Arquitectura {1}'.format(i, j + 1))
            f.write('- mse(train): ')
            f.write(str(mse_train))
            f.write('- mse(test)')
            f.write(str(mse_test))
            f.write("\n")
        print('')
        f.write("\n")
    print('*' * 16)
    print('')


users = [50, 31, 4]
number_of_architectures = 6

test_cache = pickle.load(open('pkl/test_cache.pkl', 'rb'))
train_cache = pickle.load(open('pkl/train_cache.pkl', 'rb'))
models = pickle.load(open('pkl/models.pkl', 'rb'))

test_all()


