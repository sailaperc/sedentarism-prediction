from tcn import compiled_tcn
from utils import get_train_test_data

def get_model():
    model = compiled_tcn(return_sequences=False,
                         num_feat=number_of_features,
                         num_classes=0,
                         nb_filters=6,
                         kernel_size=2,
                         dilations=[1, 2],
                         # dilations=[2 ** i for i in range(2, 5)],
                         nb_stacks=1,
                         max_len=lags,
                         use_skip_connections=True,
                         regression=True,
                         dropout_rate=0.8)
    return model


lags = 4
period = 24
epochs=64
batch=32
user = 3
metric = 'mae'
tresd = True
impersonal = True
number_of_features = 28
input_shape = (lags,number_of_features)

x_train, y_train, x_test, y_test = get_train_test_data(user,True,lags,1,'1h',True)

x_train = x_train.reshape(x_train.shape[0], lags, number_of_features)
x_test = x_test.reshape(x_test.shape[0], lags, number_of_features)
'''
input_shape = number_of_features

'''
model = get_model()
print('{0} casos de entrenamiento. **** {1} casos para testeo.'.format(x_train.shape[0], x_test.shape[0]))
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch,
                    validation_data=(x_test, y_test),
                    verbose=0)
'''
model = LinearRegression()
model.fit(x_train, y_train)

plt.close()
plt.figure(figsize=(15, 4))
plt.plot(y_train, label='Train')
y_pred = model.predict(x_train)
plt.plot(y_pred, label='Predicted')
plt.axhline(y=1.5, color='r', linestyle=':', )
plt.legend(loc='upper right')
plt.show()

#mae = mean_absolute_error(y_train, y_pred)
#print('mae:{0}'.format(mae))

plt.close()
plt.figure(figsize=(15, 4))
plt.plot(y_test, label='Test')
y_pred = model.predict(x_test)
plt.plot(y_pred, label='Predicted')
plt.axhline(y=1.5, color='r', linestyle=':', )
plt.legend(loc='upper right')
plt.show()

plt.close()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('mae:{0}'.format(mae))
'''


