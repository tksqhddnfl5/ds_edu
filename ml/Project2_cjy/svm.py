from collections import Counter
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
import pickle
import requests
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # 모든 경고 무시
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
import scipy
from scipy.signal import find_peaks
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis

def do_cv():
    def load_data():
        if Path('data/train.zip').exists():
            # Read data from local file
            with zipfile.ZipFile('data/train.zip') as zip_file:
                with zip_file.open('train.pkl') as f:
                    df = pickle.load(f)

        else:
            # Download data
            DATA_URL = 'https://gitlab.com/machine-learning-course1/ml-project-lg-2025-winter/-/raw/main/data/train.zip?ref_type=heads&inline=false'
            response = requests.get(DATA_URL)
            assert response.status_code == 200

            # Read data
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                with zip_file.open('train.pkl') as f:
                    df = pickle.load(f)

        # Convert to numpy arrays
        X, Y = np.stack(df['Data']), np.stack(df['Motion'])  # np.stack : pandas.Series => np.ndarray
        groups = np.stack(df['Subject'])  # For LeaveOneGroupOut

        return X, Y, groups

    X, Y, groups = load_data()

    doitnow = False # 지금 변수선택을 할 것인지 말 것인지. 이미 다 해 놨으니까 False로 합시다.
    if doitnow:
        from variableselection_cjy import select_variable
        select_variable()

    with open("indbwd.txt", "r") as f:
        indbwd = [int(line.strip()) for line in f]  
    with open("namebwd.txt", "r") as f:
        namebwd = [(line.strip()) for line in f] 



    def extract_features(X_sample: np.ndarray) -> np.ndarray:
        
        assert X_sample.shape == (500, 3)

        #################### TODO: Extract more features ####################

        # Extract time domain features
        X_time = X_sample

        # Extract frequency domain features
        X_freq = np.abs(np.fft.fft(X_sample, axis=0))[1:]
        dominant_freq = np.argmax(X_freq)
        theta = np.angle(np.fft.fft(X_sample, axis=0))
        compx = np.exp(1j * theta)
        thetabar = np.angle(np.mean(compx, axis=0))
        jerk = np.diff(X_time, axis=0)
        jerk_count = np.sum(np.abs(jerk) > 2)
        freqs = np.fft.fftfreq(len(X_sample))
        power_spectrum = np.abs(np.fft.fft(X_sample, axis=0))**2
        low_freq_energy = np.sum(power_spectrum[freqs < freqs.mean()], axis=0)
        high_freq_energy = np.sum(power_spectrum[freqs > freqs.mean()], axis=0)
        energy_ratio = low_freq_energy / high_freq_energy
        R = np.corrcoef(X_sample.T)
        norm = np.sqrt(np.sum(X_time**2, axis=1))
        absdiffs = np.abs(np.apply_along_axis(np.diff, axis=0, arr=X_time))
        acfs = np.apply_along_axis(lambda x: acf(x.reshape(-1), nlags=5)[1:], axis=0, arr=X_sample)

        # Concatenate features
        X_features = np.hstack([
            np.mean(X_time, axis=0), # sample mean
            np.std(X_time, ddof=1, axis=0), # sample standard deviation
            np.median(X_time, axis=0), # sample median
            np.max(X_time, axis=0), # sample maximun
            np.min(X_time, axis=0), # sample minimum
            (np.abs(X_time)).sum(axis=0),
            np.sqrt((X_time**2).sum()),
            skew(X_time, axis=0),
            kurtosis(X_time, axis=0),
            np.percentile(X_time, q=0.1, axis=0),
            # np.percentile(X_time, q=0.9, axis=0),
            np.percentile(X_time, q=0.75, axis=0) - np.percentile(X_time, q=0.25, axis=0),
            dominant_freq, # dominant frequency
            np.max(X_freq, axis=0),
            entropy(X_freq),
            np.sum(X_freq ** 2, axis=0),
            thetabar, # angle mean
            np.apply_along_axis(lambda x: len(scipy.signal.find_peaks(x)[0]), axis=0, arr=X_time), # 피크 개수
            absdiffs.mean(axis=0), # 차분 절댓값 평균크기
            np.median(absdiffs, axis=0), # 차분 절댓값 중간값크기
            np.max(absdiffs, axis=0), # 차분 절댓값 최댓값
            (absdiffs.reshape(-1) > 1).sum(),
            (absdiffs.reshape(-1) > 3).sum(),
            (absdiffs.reshape(-1) > 5).sum(),
            np.argmax(absdiffs),
            np.argmin(absdiffs),
            np.percentile(absdiffs, 0.75) - np.percentile(absdiffs, 0.25),
            np.apply_along_axis(lambda x: (x[-1] - x[0])/500, axis=0, arr=X_time), # 기울기
            np.mean(np.diff(np.where(X_time > np.mean(X_time)))), # 신호 주기성
            np.mean(np.diff(np.sign(X_time), axis=0) != 0, axis=0), # zero-crossing rate
            jerk_count,
            energy_ratio,
            R[0,1], R[0,2], R[1,2],
            acfs[:,0], # X좌표의 acf lag 1~5까지
            acfs[:,1], # Y좌표의 acf lag 1~5까지
            acfs[:,2], # Z좌표의 acf lag 1~5까지
            X_freq.mean(axis=0), # freqmean
            # X_freq.std(axis=0,ddof=1), # freqstd
            np.percentile(X_freq, 0.75, axis=0) - np.percentile(X_freq, 0.25, axis=0),
            norm.mean(),
            norm.std(ddof=1),
            norm.max(),
            np.percentile(norm, 0.75) - np.percentile(norm, 0.25),

        ])
        X_features = X_features[indbwd]

        ######################################################################

        assert X_features.ndim == 1
        return X_features

    from joblib import Parallel, delayed

    multi = True  # 병렬 처리 활성화

    if multi:
        X_features = np.array(Parallel(n_jobs=-1)(
            delayed(extract_features)(x) for x in tqdm(X, total=len(X))
        ))
    else:
        X_features = np.array([extract_features(x) for x in tqdm(X, total=len(X))])
        
    import optuna
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA

    def objective(trial):
        # 하이퍼파라미터 탐색 범위 정의
        C = trial.suggest_float("C", 10**(-4), 5)
        gamma = trial.suggest_float('gamma', 0.001, 5)

        SVMClassifier = Pipeline([
        ('scaler', StandardScaler()),
        ('preprocessor', PCA()),
        ('classifier', SVC(
            C=C,
            gamma = gamma
        )),
        ])

        # 교차 검증 점수로 평가
        scores = cross_val_score(SVMClassifier, X_features, Y, groups=groups, cv=LeaveOneGroupOut(), scoring="f1_macro")
        return scores.mean()

    # Optuna 스터디 생성 및 최적화
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=-1)

    import json
    with open("svm.txt", "w") as f:
        json.dump(study.best_params, f)
        
if __name__ == '__main__':
    do_cv()