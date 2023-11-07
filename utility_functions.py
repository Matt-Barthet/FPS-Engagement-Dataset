import numpy as np
from scipy import stats
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from krippendorff import alpha

"""def sda(trace1, trace2):
  N = min(len(trace1),len(trace2))
  trace1, trace2 = trace1[:N], trace2[:N]
  sda = 0
  for i in range(1, N):
      p = trace1[i] - trace1[i-1]
      q = trace2[i] - trace2[i-1]
      p = np.sign(p)
      q = np.sign(q)
      if p == q:
          sda += 1
      else:
          sda -= 1
  sda = sda / (N - 1)
  return sda"""

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

    if plot:
      plot_trace(trace1, trace2, sda)
    return sdas


import numpy as np
import matplotlib.pyplot as plt

def plot_trace(trace1, trace2, sda):
    plt.figure()
    plt.plot(range(len(trace1)), trace1)
    plt.plot(range(len(trace2)), trace2)
    
    for i in range(1, len(sda)):
        color = 'green' if sda[i] >= 0 else 'red'
        plt.fill_between([i-1, i], trace1[i-1:i+1], trace2[i-1:i+1], 
                         color=color, alpha=0.3)

    green_patch = plt.Rectangle((0,0),1,1,fc="green", edgecolor = 'none', alpha=0.3)
    red_patch = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none', alpha=0.3)
    plt.legend([green_patch, red_patch], ['Agree', 'Disagree'], loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()



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
