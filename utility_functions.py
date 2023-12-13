import numpy as np
from scipy import stats
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from krippendorff import alpha
from dtw import dtw
import numpy as np


def avgfilter(signal, N):
    if signal.shape == ():
        return signal
    output = np.empty_like(signal, dtype=np.float64)
    for k in range(1, N + 1):
        output[k - 1] = np.mean(signal[:k])
    for i in range(N, len(signal)):
        output[i] = np.mean(signal[i - N:i])
    return output


def get_max_times(participants):
    game_max_times = {}
    for _, participant_df in participants:
        games = participant_df.groupby('DatabaseName')
        for _, game_df in games:
            clean_game_name = game_df['OriginalName'].values[0].split("_")[1].split(".")[0]
            if clean_game_name not in game_max_times or game_df["VideoTime"].max() > game_max_times[clean_game_name]:
                game_max_times[clean_game_name] = game_df["VideoTime"].max()
    return game_max_times


def dtw_distance(signal1, signal2):
  return dtw(signal1, signal2).distance

def sda(trace1, trace2, plot=False):
    N = min(len(trace1),len(trace2))
    trace1, trace2 = trace1[:N], trace2[:N]
    sda = [0]
    for i in range(1, N):
        p = np.sign(trace1[i] - trace1[i-1])
        q = np.sign(trace2[i] - trace2[i-1])
        if p == q:
            sda.append(1)
        else:
            sda.append(-1)
    sdas = np.sum(sda) / (N-1)
    return sdas

def count_changes(arr):
    arr = np.asarray(arr)
    if len(arr) < 2:
        return 0
    change_count = 0
    for i in range(len(arr) - 1):
        if arr[i] != arr[i + 1]:
            change_count += 1
    return change_count

def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)  # Ensure the input data is a numpy array
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)  # Margin of error
    return np.round(mean, 4), np.round(ci, 4)

def pairwise_correlation(data, corr_function):
  correlations = []
  unique_pairs = list(combinations(data, 2))
  for signal1, signal2 in unique_pairs:
      correlations.append(corr_function(signal1, signal2))
  return compute_confidence_interval(correlations)


def compute_cohens_kappa(signal1, signal2):
    N = min(len(signal1),len(signal2))
    signal1, signal2 = signal1[:N], signal2[:N]
    ordinal_signal1 = np.sign(np.diff(signal1))
    ordinal_signal2 = np.sign(np.diff(signal2))
    ordinal_signal1 = ordinal_signal1 + 1
    ordinal_signal2 = ordinal_signal2 + 1
    kappa = cohen_kappa_score(ordinal_signal1, ordinal_signal2)
    return kappa

def compute_krippendorffs_alpha(data):
  N = np.inf
  for signal in data:
    N = min(N, len(signal))
  for i in range(len(data)):
    data[i] = data[i][:N]
  if not isinstance(data, np.ndarray):
      data = np.array(data)
  alpha_value = alpha(data)
  return alpha_value

def cronbach_alpha(data):
  N = np.inf
  for signal in data:
    N = min(N, len(signal))
  for i in range(len(data)):
    data[i] = data[i][:N]
  if not isinstance(data, np.ndarray):
      data = np.array(data)
  num_items = data.shape[1]
  item_variances = np.var(data, axis=0, ddof=1)
  total_variance = np.var(data.sum(axis=1), ddof=1)
  alpha = (num_items / (num_items - 1)) * (1 - (item_variances.sum() / total_variance))
  return alpha
