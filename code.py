#%%
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model

from tcn import TCN, tcn_full_summary
import numpy as np

batch_size, timesteps, input_dim = None, 20, 2





i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(nb_filters=4, return_sequences=False, dilations=(1,2))(i)  # The TCN layers are here.
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')
#%%
tcn_full_summary(m, expand_residual_blocks=True)
#%%
x, y = get_x_y()
m.fit(x, y, epochs=2, validation_split=0.2)


#%%
from utils.utils_graphic import plot_user_selection
plot_user_selection(2)

# %%
