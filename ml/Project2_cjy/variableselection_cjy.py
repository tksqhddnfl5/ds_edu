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

def select_variable():

    def load_data():
        if Path('data/train.zip').exists():
            # Read data from local file
            print('Reading data...')
            with zipfile.ZipFile('data/train.zip') as zip_file:
                with zip_file.open('train.pkl') as f:
                    df = pickle.load(f)

        else:
            # Download data
            print('Downloading data...')
            DATA_URL = 'https://gitlab.com/machine-learning-course1/ml-project-lg-2025-winter/-/raw/main/data/train.zip?ref_type=heads&inline=false'
            response = requests.get(DATA_URL)
            assert response.status_code == 200

            # Read data
            print('Reading data...')
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                with zip_file.open('train.pkl') as f:
                    df = pickle.load(f)

        # Convert to numpy arrays
        X, Y = np.stack(df['Data']), np.stack(df['Motion'])  # np.stack : pandas.Series => np.ndarray
        groups = np.stack(df['Subject'])  # For LeaveOneGroupOut

        return X, Y, groups

    X, Y, groups = load_data()

    print('Data have been loaded!!!')

    # 데이터 예시 (실제 데이터에 맞게 사용)
    categories = np.unique(Y)  # Y의 고유한 행동 범주 추출
    num_categories = len(categories)

    import scipy
    from scipy.signal import find_peaks
    from scipy.stats import entropy
    from statsmodels.tsa.stattools import acf
    from scipy.stats import skew, kurtosis

    def extract_features(X_sample: np.ndarray, ind = -1) -> np.ndarray:
        """
        Extract features from a single sample

        Parameters
        ----------
        X_sample : array of shape (500, 3)
            100Hz * 5 seconds => 500
            3 axis (x, y, z)  => 3

        Returns
        -------
        features : array with (p,) shape
            Extracted features from X_sample
        """
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
        if ind != -1:
            X_features = X_features[ind]

        ######################################################################

        assert X_features.ndim == 1
        return X_features

    names = [
        'meanX','meanY','meanZ','sdX','sdY','sdZ','medX','medY','medZ','maxX','maxY','maxZ','minX','minY','minZ','abssumX','abssumY','abssumZ','sqsum','skewX','skewY','skewZ','kurtX','kurtY','kurtZ',
        'per10X','per10Y','per10Z',
        # 'per90X','per90Y','per90Z',
        'iqrX','iqrY','iqrZ',
        'argdomi','domfreqX','domfreqY','domfreqZ','entropyX','entropyY','entropyZ','sqsumfreqX','sqsumfreqY','sqsumfreqZ','anglemeanX','anglemeanY','anglemeanZ',
        'numpeakX','numpeakY','numpeakZ',
        'absdiffmeanX','absdiffmeanY','absdiffmeanZ',
        'absdiffmedX','absdiffmedY','absdiffmedZ','absdiffmaxX','absdiffmaxY','absdiffmaxZ',
        'diffbigger1','diffbigger3','diffbigger5','diffargmax','diffargmin','absdiffiqr',
        'betax','betay','betaz','sig','zcrX','zcrY','zcrZ','zerk_count','enerratioX','enerratioY','enerratioZ',
        'corrXY','corrXZ','corrYZ','acfX1','acfX2','acfX3','acfX4','acfX5','acfY1','acfY2','acfY3','acfY4','acfY5','acfZ1','acfZ2','acfZ3','acfZ4','acfZ5',
        'freqmeanX','freqmeanY','freqmeanZ','freqiqrX','freqiqrY','freqiqrZ',
        'normmean','normstd','normmax','normiqr',
    ]

    from joblib import Parallel, delayed

    multi = True  # 병렬 처리 활성화

    if multi:
        X_features = np.array(Parallel(n_jobs=-1)(
            delayed(extract_features)(x) for x in tqdm(X, total=len(X))
        ))
    else:
        X_features = np.array([extract_features(x) for x in tqdm(X, total=len(X))])
        
    X_features_o = X_features

    ### 주요 테크닉: 단 50%의 샘플만 수집하여 후진 선택법을 합니다.
    # 초기엔 계산 시간을 줄이려고 한 것이지만, 이것이 더 나은 성능을 가져다준다는 것을 경험적으로 알게 됨.
    np.random.seed(100)
    ind50 = np.random.choice(X_features_o.shape[0], 
                             size=int(X_features_o.shape[0] * 0.5), 
                             replace=False)
    X_features_o50 = X_features_o[ind50]
    Y50 = Y[ind50]
    groups50 = groups[ind50]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearDiscriminantAnalysis()),
    ])
    fwd = SequentialFeatureSelector(model, forward = False, 
                                    scoring='f1_macro',
                                    k_features = 'best',
                                    verbose=2, n_jobs=-1,
                                    cv=LeaveOneGroupOut(),
                                    # cv=5
                                    )
    # fwd.fit(X_features_o, Y, groups=groups)
    print("Now it's time to select variables with backward stepwise selection!!!!!")
    with parallel_backend("loky"):  # CPU 병렬 처리
        fwd.fit(X_features_o50, Y50, groups=groups50)

    indbwd = list(fwd.k_feature_idx_)
    namebwd = np.array(names)[indbwd]
    with open("indbwd.txt", "w") as f:
        for num in indbwd:
            f.write(f"{num}\n")  # 한 줄에 하나씩 저장
    with open("namebwd.txt", "w") as f:
        for num in list(namebwd):
            f.write(f"{num}\n")  # 한 줄에 하나씩 저장

if __name__ == "__main__":
    select_variable()