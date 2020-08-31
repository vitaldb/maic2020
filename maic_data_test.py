import pandas as pd
import numpy as np
import os
import csv
from sklearn.utils import shuffle

# 2초 moving average
def moving_average(a, n=200):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

SRATE = 100
MINUTES_AHEAD = 5
MAX_CASES = 300
TEST_NAME = 'test1'

# 전체 art 케이스를 로딩
# 제외 기준, 포함 기준은 다 포함 되어있음
df_cases = pd.read_excel("vitaldb_test_cases.xlsx")
df_cases = shuffle(df_cases)

# 최종 데이터셋 출력 파일 생성
fo = csv.writer(open('{}.csv'.format(TEST_NAME), 'w', newline=''), quoting=csv.QUOTE_MINIMAL)
row = ['hypotension', 'age', 'sex', 'weight', 'height']
for i in range(2000):
    row.append('abp_{}'.format(i+1))
fo.writerow(row)

# 각 case를 돌면서
ncase = 0
np.errstate(invalid='ignore')
for _, row in df_cases.iterrows():
    caseid = row['caseid']
    age = row['age']
    sex = row['sex']
    height = row['height']
    weight = row['weight']
    tid = row['tid']

    print('{}...'.format(caseid), flush=True, end='')

    # load wav data
    vals = pd.read_csv('https://api.vitaldb.net/' + tid, na_values=['nan','-nan(ind)']).values[:,1].astype(float)
    if len(vals) < 2 * 3600 * 500:
        print('caselen < 2')
        continue

    # 500hz -> 100hz
    R = 5
    vals = np.pad(vals, (0, R - len(vals) % R), 'constant')
    vals = vals.reshape(-1, 5)[:,0]  # resample to 100hz

    # 앞 뒤의 결측값을 제거
    case_valid_mask = ~np.isnan(vals)
    vals = vals[(np.cumsum(case_valid_mask) != 0) & (np.cumsum(case_valid_mask[::-1])[::-1] != 0)]

    if np.nanmax(vals) < 120:
        print('mbp < 120')
        continue

    # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
    i = 0
    nsamp = 0
    nevent = 0
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
            row = [event, age, sex, weight, height]
            row.extend(np.round(segx, 2))
            fo.writerow(row)
            nevent += 1
            nsamp += 1
        elif np.nanmin(segy) > 65:  # non event
            row = [event, age, sex, weight, height]
            row.extend(np.round(segx, 2))
            fo.writerow(row)
            nsamp += 1

        i += 30 * SRATE  # 30sec

    if nsamp > 0:
        print('{}: {} ({:.1f}%)'.format(caseid, nsamp, nevent * 100 / nsamp))
        ncase += 1

    if ncase >= MAX_CASES:
        break

# 데이터를 다시 읽고
df = pd.read_csv('{}.csv'.format(TEST_NAME))
df = shuffle(df)  # 무작위 배치

# 분리하여 저장
df['hypotension'].to_csv('{}_y.csv'.format(TEST_NAME), header=False, index=False)
df.drop(columns='hypotension').to_csv('{}_x.csv'.format(TEST_NAME), index=False)
