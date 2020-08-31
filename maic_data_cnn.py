import numpy as np
import pandas as pd
import os
import pickle

MINUTES_AHEAD = 5
SRATE = 100

# 2초 moving average
def moving_average(a, n=200):
    ret = np.nancumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# training set 로딩
x_train = []  # arterial waveform
y_train = []  # hypotension
if os.path.exists('x_train.npz'):
    print('loading train...', flush=True, end='')
    x_train = np.load('x_train.npz')['arr_0']
    y_train = np.load('y_train.npz')['arr_0']
    print('done', flush=True)
else:
    df_train = pd.read_csv('train_cases.csv')
    for _, row in df_train.iterrows():
        caseid = row['caseid']
        age = row['age']
        sex = row['sex']
        weight = row['weight']
        height = row['height']

        vals = pd.read_csv('train_data/{}.csv'.format(caseid), header=None).values.flatten()

        # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
        i = 0
        event_idx = []
        non_event_idx = []
        while i < len(vals) - SRATE * (20 + (1 + MINUTES_AHEAD) * 60):
            segx = vals[i:i + SRATE * 20]
            segy = vals[i + SRATE * (20 + MINUTES_AHEAD * 60):i + SRATE * (20 + (1 + MINUTES_AHEAD) * 60)]

            # 결측값 10% 이상이면
            if np.mean(np.isnan(segx)) > 0.1 or \
                np.mean(np.isnan(segy)) > 0.1 or \
                np.max(segx) > 200 or np.min(segx) < 20 or \
                np.max(segy) > 200 or np.min(segy) < 20 or \
                np.max(segx) - np.min(segx) < 30 or \
                np.max(segy) - np.min(segy) < 30 or \
                (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any() or \
                (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
                i += SRATE  # 1 sec 씩 전진
                continue

            # 출력 변수
            segy = moving_average(segy, 2 * SRATE)  # 2 sec moving avg
            event = 1 if np.nanmax(segy) < 65 else 0
            if event:  # event
                event_idx.append(i)
                x_train.append(segx)
                y_train.append(event)
            elif np.nanmin(segy) > 65:  # non event
                non_event_idx.append(i)
                x_train.append(segx)
                y_train.append(event)

            i += 30 * SRATE  # 30sec

        nsamp = len(event_idx) + len(non_event_idx)
        if nsamp > 0:
            print('{}: {} ({:.1f}%)'.format(caseid, nsamp, len(event_idx) * 100 / nsamp))

    print('saving...', flush=True, end='')
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=bool)
    np.savez_compressed('x_train.npz', x_train)
    np.savez_compressed('y_train.npz', y_train)
    print('done', flush=True)

# test set 로딩
if os.path.exists('x_test.npz'):
    print('loading test...', flush=True, end='')
    x_test = np.load('x_test.npz')['arr_0']
    print('done', flush=True)
else:
    x_test = pd.read_csv('test1_x.csv').values

    print('saving...', flush=True, end='')
    x_test = np.array(x_test[:,4:], dtype=np.float32)
    np.savez_compressed('x_test.npz', x_test)
    print('done', flush=True)

BATCH_SIZE = 512

x_train -= 65
x_train /= 65
x_test -= 65
x_test /= 65

# nan 을 이전 값으로 채움
x_train = pd.DataFrame(x_train).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
x_test = pd.DataFrame(x_test).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values

# CNN에 입력으로 넣기 위해 차원을 추가
x_train = x_train[..., None]
x_test = x_test[..., None]

print('train {} ({} events {:.1f}%), test {}'.format(len(y_train), sum(y_train), 100*np.mean(y_train), len(x_test)))

from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve
import tensorflow as tf

num_nodes = [64, 64, 64, 64, 64, 64]

testname = '-'.join([str(num_node) for num_node in num_nodes])
print(testname)

# 출력 폴더를 생성
odir = "output"
if not os.path.exists(odir):
    os.mkdir(odir)
weight_path = odir + "/weights.hdf5"

# build a model
model = Sequential()
for num_node in num_nodes:
    model.add(Conv1D(filters=num_node, kernel_size=3, padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=BATCH_SIZE, class_weight={0:1, 1:10}, 
                        callbacks=[ModelCheckpoint(monitor='val_loss', filepath=weight_path, verbose=1, save_best_only=True),
                                    EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')])

# 모델을 저장
open(odir + "/model.json", "wt").write(model.to_json())

model.load_weights(weight_path)

# 전체 test 샘플을 한번에 예측
y_pred = model.predict(x_test).flatten()

# 결과를 저장
np.savetxt('pred_y.txt', y_pred)

