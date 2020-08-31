import numpy as np
import pandas as pd
import os
import csv
import pickle
import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 100000
MINUTES_AHEAD = 5
SRATE = 100

# 2초 moving average
def moving_average(a, n=200):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# training set 로딩
if os.path.exists('x_train.npz'):
    print('loading train...', flush=True, end='')
    x_train = np.load('x_train.npz')['arr_0']
    y_train = np.load('y_train.npz')['arr_0']
    print('done', flush=True)
else:
    x_train = []
    y_train = []
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

            if False:  #len(event_idx) / nsamp > 0.01:  # 이벤트 비율이 1% 이상인 경우 그림을 그림
                vals2 = moving_average(vals, 2 * SRATE)
                
                t = np.arange(0, len(vals)) * 0.01

                # 그림 생성
                plt.figure(figsize=(20, 4))
                plt.xlim([0, max(t)])
                plt.ylim([0, 150])

                # 65 mmHg 가로선
                plt.plot(t, vals, color='k', alpha=0.1)  # 웨이브를 그린다
                plt.plot(t[:len(vals2)], vals2, color='y', alpha=0.5)  # trend를 그린다
                plt.axhline(y=65, color='r', alpha=0.5)

                # 저혈압 상태일 때를 붉은 반투명 배경으로
                for i in event_idx:
                    plt.axvspan(i / SRATE + MINUTES_AHEAD * 60, (i + 1) / SRATE + MINUTES_AHEAD * 60, color='r', alpha=0.5, lw=1)
                for i in non_event_idx:
                    plt.axvspan(i / SRATE + MINUTES_AHEAD * 60, (i + 1) / SRATE + MINUTES_AHEAD * 60, color='b', alpha=0.5, lw=1)
                
                # 그림 저장
                plt.savefig('extract/{}.png'.format(caseid))
                plt.close()

    print('saving...', flush=True, end='')
    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=bool)
    np.savez_compressed('x_train.npz', x_train)
    np.savez_compressed('y_train.npz', y_train)
    print('done', flush=True)

# test set 로딩
if os.path.exists('x_test.npz'):
    print('loading test...', flush=True, end='')
    x_test = np.load('x_test.npz')['arr_0']
    y_test = np.load('y_test.npz')['arr_0']
    print('done', flush=True)
else:
    x_test = pd.read_csv('test1_x.csv').values[:,4:]
    y_test = pd.read_csv('test1_y.csv', header=None).values.flatten()

    print('saving...', flush=True, end='')
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=bool)
    np.savez_compressed('x_test.npz', x_test)
    np.savez_compressed('y_test.npz', y_test)
    print('done', flush=True)

tests = [
    # [4],
    # [6],
    # [8],
    [64, 64],
    [64, 64, 64],
    [64, 64, 64, 64],
    [64, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 64],
    # [4,4,4,4],
    # [8,8,8,8],
    # [16,16,16,16],
    # [32,32,32,32],
    # [16,6],
    # [4,4,4],
    # [6,6,6],
]

BATCH_SIZE = 256

x_train -= 65
x_train /= 65
x_test -= 65
x_test /= 65

# nan 을 이전 값으로 채움
x_train = pd.DataFrame(x_train).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
x_test = pd.DataFrame(x_test).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values

# 2000 sample --> 10 sample
# x_train = np.nanmean(x_train.reshape(x_train.shape[0], 10, -1), axis=2)
# x_test = np.nanmean(x_test.reshape(x_test.shape[0], 10, -1), axis=2)

# CNN에 입력으로 넣기 위해 차원을 추가
x_train = x_train[..., None]
x_test = x_test[..., None]

print('train {} ({} events {:.1f}%), test {} ({} events {:.1f}%)'.format(len(y_train), sum(y_train), 100*np.mean(y_train), len(y_test), sum(y_test), 100*np.mean(y_test)))

from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import tensorflow as tf

for rnodes in tests:
    testname = '-'.join([str(rnode) for rnode in rnodes])
    print(testname)

    # 출력 폴더를 생성
    odir = "output"
    if not os.path.exists(odir):
        os.mkdir(odir)
    weight_path = odir + "/weights.hdf5"

    # build a model
    model = Sequential()
    for i in range(len(rnodes)):
        rnode = rnodes[i]
        if i == 0:
            model.add(Conv1D(filters=rnode, kernel_size=3, padding='valid'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D())
        else:
            model.add(Conv1D(filters=rnode, kernel_size=3, padding='valid'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D())
    
    model.add(GlobalMaxPool1D())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
    hist = model.fit(x_train, y_train, validation_split=0.1, epochs=100, batch_size=BATCH_SIZE, class_weight={0:1, 1:5},    
                            callbacks=[ModelCheckpoint(monitor='val_loss', filepath=weight_path, verbose=1, save_best_only=True),
                                        EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')])

    # 모델을 저장
    open(odir + "/model.json", "wt").write(model.to_json())

    model.load_weights(weight_path)

    # 전체 test 샘플을 한번에 예측
    y_pred = model.predict(x_test).flatten()
    test_err = y_pred - y_test
    test_rmse = np.mean(np.square(test_err)) ** 0.5

    precision, recall, thvals = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)

    fpr, tpr, thvals = roc_curve(y_test, y_pred)
    auroc = auc(fpr, tpr)
    
    thval = 0.5
    f1 = f1_score(y_test, y_pred > thval)
    acc = accuracy_score(y_test, y_pred > thval)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred > thval).ravel()

    # print results
    print("Test RMSE: {}".format(test_rmse))
    res = 'auroc={:.3f}, auprc={:.3f} acc={:.3f}, F1={:.3f}, PPV={:.1f}, NPV={:.1f}, TN={}, fp={}, fn={}, TP={}'.format(auroc, auprc, acc, f1, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp)
    print(res)

    # save auc curve
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('{}/auroc.png'.format(odir))
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('{}/auprc.png'.format(odir))
    plt.close()

    os.rename(odir, testname + ' ' + res)
