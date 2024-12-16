import json
import numpy as np
import pandas as pd
import PIL.Image as PImage
import string
import urllib.request as request

from numpy.linalg import det as np_det, inv as np_inv
from os import listdir, path
from random import seed, shuffle

from sklearn.cluster import KMeans as SklKMeans, SpectralClustering as SklSpectralClustering
from sklearn.decomposition import PCA as SklPCA
from sklearn.ensemble import RandomForestClassifier as SklRandomForestClassifier
from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, root_mean_squared_error
from sklearn.mixture import GaussianMixture as SklGaussianMixture
from sklearn.preprocessing import MinMaxScaler as SklMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklStandardScaler
from sklearn.preprocessing import PolynomialFeatures as SklPolynomialFeatures
from sklearn.svm import SVC as SklSVC

from warnings import simplefilter


def object_from_json_url(url):
  with request.urlopen(url) as in_file:
    return json.load(in_file)


def regression_error(labels, predicted):
  if not (isinstance(labels, pd.core.frame.DataFrame) or isinstance(labels, pd.core.series.Series)):
    raise Exception("truth labels has wrong type. Please use pandas DataFrame or Series")
  if not (isinstance(predicted, pd.core.frame.DataFrame) or isinstance(predicted, pd.core.series.Series)):
    raise Exception("predicted labels has wrong type. Please use pandas DataFrame or Series")

  return root_mean_squared_error(labels.values, predicted.values)


def classification_error(labels, predicted):
  if not (isinstance(labels, pd.core.frame.DataFrame) or isinstance(labels, pd.core.series.Series)):
    try:
      labels = pd.DataFrame(labels)
    except:
      raise Exception("truth labels has wrong type. Please use pandas DataFrame or Series")
  if not (isinstance(predicted, pd.core.frame.DataFrame) or isinstance(predicted, pd.core.series.Series)):
    try:
      predicted = pd.DataFrame(predicted)
    except:
      raise Exception("predicted labels has wrong type. Please use pandas DataFrame or Series")

  return 1.0 - accuracy_score(labels.values, predicted.values)


def display_confusion_matrix(labels, predicted, display_labels):
  simplefilter(action='ignore', category=FutureWarning)
  ConfusionMatrixDisplay.from_predictions(labels, predicted, display_labels=display_labels, xticks_rotation="vertical")


class PolynomialFeatures(SklPolynomialFeatures):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit_transform(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")

    self.columns = X.columns
    self.shape = X.shape
    X_t = super().fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=self.get_feature_names_out())

  def transform(self, X, *args, **kwargs):
    if type(X) == np.ndarray:
      return super().transform(X, *args, **kwargs)

    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = super().transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=self.get_feature_names_out())


class Predictor():
  def __init__(self, type, **kwargs):
    if type == "linear":
      self.model = SklLinearRegression(**kwargs)
    elif type == "forest":
      if "max_depth" not in kwargs:
        kwargs["max_depth"] = 16
      self.model = SklRandomForestClassifier(**kwargs)
    elif type == "svc":
      if "kernel" not in kwargs:
        kwargs["kernel"] = "linear"
      self.model = SklSVC(**kwargs)

  def fit(self, X, y, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    if not (isinstance(y, pd.core.frame.DataFrame) or isinstance(y, pd.core.series.Series)):
      raise Exception("Label input has wrong type. Please use pandas DataFrame or Series")

    self.y_name = y.name if len(y.shape) == 1 else y.columns[0]
    self.model.fit(X.values, y.values, *args, **kwargs)
    return self

  def predict(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    y_t = self.model.predict(X.values, *args, **kwargs)
    return pd.DataFrame(y_t, columns=[self.y_name])


class Scaler():
  def __init__(self, type, **kwargs):
    if type == "minmax":
      self.scaler = SklMinMaxScaler(**kwargs)
    elif type == "std":
      self.scaler = SklStandardScaler(**kwargs)

  def fit_transform(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")

    self.columns = X.columns
    self.shape = X.shape
    X_t = self.scaler.fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def transform(self, X, *args, **kwargs):
    if type(X) == np.ndarray:
      return self.scaler.transform(X, *args, **kwargs)

    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")
    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = self.scaler.transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def inverse_transform(self, X, *args, **kwargs):
    if not (isinstance(X, pd.core.frame.DataFrame) or isinstance(X, pd.core.series.Series)):
      raise Exception("Input has wrong type. Please use pandas DataFrame or Series")

    col = ""
    col_vals = []

    if len(X.shape) == 1:
      col = X.name
      col_vals = X.values
    elif len(X.shape) == 2 and X.shape[1] == 1:
      col = X.columns[0]
      col_vals = X[col].values

    if col != "":
      X = pd.DataFrame(X.values, columns=[col])
      dummy_df = pd.DataFrame(np.zeros((len(col_vals), self.shape[1])), columns=self.columns)
      dummy_df[col] = col_vals
      X_t = self.scaler.inverse_transform(dummy_df.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=self.columns)[[col]]

    else:
      X_t = self.scaler.inverse_transform(X.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=X.columns)


class Clusterer():
  def __init__(self, type, **kwargs):
    self.num_clusters = 0
    kwargs["n_init"] = 10
    if type == "kmeans":
      self.model = SklKMeans(**kwargs)
    elif type == "gaussian":
      if "n_clusters" in kwargs:
        kwargs["n_components"] = kwargs["n_clusters"]
        del kwargs["n_clusters"]
      self.model = SklGaussianMixture(**kwargs)
    elif type == "spectral":
      if "affinity" not in kwargs:
        kwargs["affinity"] = 'nearest_neighbors'
      if "n_clusters" in kwargs:
        kwargs["n_clusters"] += 0
      self.model = SklSpectralClustering(**kwargs)

  def fit_predict(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")

    y = self.model.fit_predict(X.values, *args, **kwargs)
    self.X = X.values
    self.y = y
    self.num_clusters = len(np.unique(y))
    self.num_features = self.X.shape[1]
    self.cluster_centers_ = np.array([self.X[self.y == c].mean(axis=0) for c in range(self.num_clusters)]).tolist()
    return pd.DataFrame(y, columns=["clusters"])

  def distance_error(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")

    point_centers = [self.cluster_centers_[i] for i in self.y]
    point_diffs = np.array([p - c for p,c in zip(self.X, point_centers)])

    cluster_L2 = [np.sqrt(np.square(point_diffs[self.y == c]).sum(axis=1)).mean() for c in range(self.num_clusters)]

    return sum(cluster_L2) / len(cluster_L2)

  def likelihood_error(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")

    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
    means = np.array([self.X[self.y == c].mean(axis=0) for c in range(self.num_clusters)])
    covs = np.array([np.cov(self.X[self.y==c].transpose()) for c in range(self.num_clusters)])

    point_means = [means[i] for i in self.y]
    point_covs = [covs[i] for i in self.y]

    two_pi_term = np.power(2 * np.pi, self.num_features)

    point_density_den = [np.sqrt(two_pi_term * np_det(cov)) for cov in point_covs]
    point_density_num = [np.exp(-0.5 * (p - m) @ np_inv(cov) @ (p - m)) for p,m,cov in zip(self.X, point_means, point_covs)]
    point_density = np.array(point_density_num) / np.array(point_density_den)

    cluster_log_like = [np.log(point_density[self.y == c]).mean() for c in range(self.num_clusters)]

    return sum(cluster_log_like) / len(cluster_log_like)

  def balance_error(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")
    counts = np.unique(self.y, return_counts=True)[1]
    sum_dists = np.abs(counts / len(self.y) - (1 / self.num_clusters)).sum()
    scale_factor = 0.5 * self.num_clusters / (self.num_clusters - 1)
    return scale_factor * sum_dists


class PCA(SklPCA):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.pc_labels = []
    self.o_labels = None

  def check_input(self, X):
    if isinstance(X, pd.core.frame.DataFrame):
      self.o_labels = X.columns
      X = X.values.tolist()
    if not isinstance(X, list):
      raise Exception("Input has wrong type. Please use list of list of pixels")
    if not isinstance(X[0], list):
      raise Exception("Input has wrong type. Please use list of list of pixels")
    return X

  def fit(self, X, *args, **kwargs):
    X = self.check_input(X)
    super().fit(X, *args, **kwargs)
    self.pc_labels = [f"PC{i}" for i in range(self.n_components_)]

  def transform(self, X, *args, **kwargs):
    if len(self.pc_labels) != self.n_components_:
      raise Exception("Error: need to run fit() first")
    X = self.check_input(X)
    X_t = super().transform(X, *args, **kwargs)
    X_obj = [{f"PC{i}": v for i,v in enumerate(x)} for x in X_t]
    return pd.DataFrame.from_records(X_obj)

  def fit_transform(self, X, *args, **kwargs):
    self.fit(X, *args, **kwargs)
    return self.transform(X, *args, **kwargs)

  def inverse_transform(self, X_t, *args, **kwargs):
    if not (isinstance(X_t, pd.core.frame.DataFrame) or isinstance(X_t, pd.core.series.Series)):
      raise Exception("Input has wrong type. Please use pandas DataFrame or Series")
    if len(self.pc_labels) != self.n_components_:
      raise Exception("Error: need to run fit() first")

    X_t_np = X_t[self.pc_labels].values
    if isinstance(X_t, pd.core.frame.DataFrame) and X_t_np.shape[1] != self.n_components_:
      raise Exception("Input has wrong shape. Check number of features")
    if isinstance(X_t, pd.core.series.Series) and X_t_np.shape[0] != self.n_components_:
      raise Exception("Input has wrong shape. Check number of features")

    X_i_np = super().inverse_transform(X_t_np)
    return pd.DataFrame(X_i_np, columns=self.o_labels)

  def explained_variance(self):
    if len(self.pc_labels) != self.n_components_:
      raise Exception("Error: need to run fit() first")
    return sum(self.explained_variance_ratio_)


class LinearRegression(Predictor):
  def __init__(self, **kwargs):
    super().__init__("linear", **kwargs)

class RandomForestClassifier(Predictor):
  def __init__(self, **kwargs):
    super().__init__("forest", **kwargs)

class SVC(Predictor):
  def __init__(self, **kwargs):
    super().__init__("svc", **kwargs)

class MinMaxScaler(Scaler):
  def __init__(self, **kwargs):
    super().__init__("minmax", **kwargs)

class StandardScaler(Scaler):
  def __init__(self, **kwargs):
    super().__init__("std", **kwargs)

class KMeansClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("kmeans", **kwargs)

class GaussianClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("gaussian", **kwargs)

class SpectralClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("spectral", **kwargs)


class LFWUtils:
  FACE_IMAGES = "./data/images/lfw/cropped"
  if not path.isdir(FACE_IMAGES):
    FACE_IMAGES = FACE_IMAGES.replace("/images/lfw/", "/image/lfw/")

  FACE_IMAGES_DIRS = sorted(listdir(FACE_IMAGES)) if path.isdir(FACE_IMAGES) else []

  LABELS = [d.split("-")[0] for d in FACE_IMAGES_DIRS if d[0] in string.ascii_letters]
  L2I = {v:i for i,v in enumerate(LABELS)}

  if len(FACE_IMAGES_DIRS) > 0:
    FACE_IMAGES_DIRS_PATH = path.join(FACE_IMAGES, FACE_IMAGES_DIRS[0])
    _first_img = [f for f in listdir(FACE_IMAGES_DIRS_PATH) if f.endswith(".jpeg") or f.endswith(".jpg")][0]
    IMAGE_SIZE = PImage.open(path.join(FACE_IMAGES_DIRS_PATH, _first_img)).size

  @staticmethod
  def train_test_split(test_pct=0.5, random_state=101010):
    seed(random_state)
    dataset = { k : { "pixels": [], "labels": [], "files": [] } for k in ["test", "train"] }
    label_files = { k : [] for k in dataset.keys() }

    for label in LFWUtils.LABELS:
      label_path = path.join(LFWUtils.FACE_IMAGES, label)
      label_files_all = [f for f in listdir(label_path) if f.endswith(".jpeg") or f.endswith(".jpg")]
      shuffle(label_files_all)
      split_idx = int(test_pct * len(label_files_all))
      label_files["test"] = label_files_all[:split_idx]
      label_files["train"] = label_files_all[split_idx:]

      for split in dataset.keys():
        for f in label_files[split]:
          img = PImage.open(path.join(label_path, f))
          img.pixels = list(img.getdata())

          pixel = img.pixels[0]
          if (type(pixel) == list or type(pixel) == tuple) and len(pixel) > 2:
            img.pixels = [sum(l[:3]) / 3 for l in img.pixels]

          dataset[split]["pixels"].append(img.pixels)
          dataset[split]["labels"].append(LFWUtils.L2I[label])
          dataset[split]["files"].append(f)

    return dataset["train"], dataset["test"]

  @staticmethod
  def top_precision(labels, predicted, top=5):
    labels_np = np.array(LFWUtils.LABELS)
    cm = confusion_matrix(labels, predicted)
    precision_sum = np.sum(cm, axis=0)
    precision = [c/t if t != 0 else 0 for c,t in zip(np.diagonal(cm), precision_sum)]
    top_idx = np.argsort(precision)
    top_precision = list(reversed(labels_np[top_idx]))
    return top_precision[:top]

  @staticmethod
  def top_recall(labels, predicted, top=5):
    labels_np = np.array(LFWUtils.LABELS)
    cm = confusion_matrix(labels, predicted)
    recall_sum = np.sum(cm, axis=1)
    recall = [c/t if t != 0 else 0 for c,t in zip(np.diagonal(cm), recall_sum)]
    top_idx = np.argsort(recall)
    top_recall = list(reversed(labels_np[top_idx]))
    return top_recall[:top]
