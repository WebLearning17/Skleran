import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, accuracy_score
from keras.layers import Masking

model = Sequential()

data_path = '../data/'

y = np.load(data_path + 'label.npy')
# X_egemaps = np.load(data_path + 'audio_features_egemaps_all.npy')[:, 1:]
# X_mfcc = np.load(data_path + 'audio_features_mfcc_all.npy')[:, 1:]
X_xbow = np.load(data_path + 'audio_features_xbow_all.npy')[:, 1:]
# l_egemaps = np.load(data_path + 'frame_num_egemaps.npy')
# l_mfcc = np.load(data_path + 'frame_num_mfcc.npy')
l_xbow = np.load(data_path + 'frame_num_xbow.npy')


def sample_frame(data):
    idata = [data[0]]
    for i in range(1, (data.shape[0]-1)//1+1):
        idata.append(data[i*1])
    return idata

timesteps = 100

def lstm_data(data, num):
    sum = 0
    ldata = []
    mnum = num.max()//1 + 1
    print(mnum)
    for i in range(len(num)):
        tmpdata = sample_frame(data[sum:sum+num[i], :])
        tmp = np.zeros((mnum - len(tmpdata), data.shape[1]))
        tmpdata = tmpdata + tmp.tolist()
        if ldata == []:
            ldata = [tmpdata]
        else:
            ldata.append(tmpdata)
        sum = sum+num[i]
    ldata = np.array(ldata)[:, :100, :]
    print(ldata.shape)
    return ldata


# X = lstm_data(X_egemaps, l_egemaps)
X = lstm_data(X_xbow, l_xbow)
y = to_categorical(y, num_classes=None)

X_train = X[:4917, :]
y_train = y[:4917, :]
X_test = X[4917:, :]
y_test = y[4917:, :]


class MAP_eval(keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.v = validation_data
        self.maps = []

    def eval_map(self):
        x_val, y_true = self.v
        y_pred = self.model.predict(x_val)
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return precision_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)

    def on_epoch_end(self, epoch, logs={}):
        map, acc = self.eval_map()
        self.maps.append([map, acc])
        print(epoch, map, acc)



num_classes = 8
data_dim = X.shape[-1]
model.add(Masking(mask_value=0., input_shape=(timesteps, data_dim)))
# model.add(LSTM(32, return_sequences=True,
#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=False, dropout=0.2))  # returns a sequence of vectors of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

map_eval = MAP_eval((X_test, y_test))
hist = model.fit(X_train, y_train, batch_size=256, verbose=2, callbacks=[keras.callbacks.TerminateOnNaN(), map_eval], epochs=80)

print(max(map_eval.maps))
