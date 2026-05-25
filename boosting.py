from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class TargetEncoder:

    def __init__(self, cat_features: Iterable[int] | None = None, ordered: bool = False, prior_weight: float = 1.0):
        self.cat_features = [] if cat_features is None else list(cat_features)
        self.ordered = ordered
        self.prior_weight = prior_weight
        self.maps_: dict[int, dict[object, float]] = {}
        self.global_mean_: float = 0.5
        self.train_encoded_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_obj = np.asarray(X, dtype=object)
        y01 = (np.asarray(y) == 1).astype(float)
        self.global_mean_ = float(y01.mean()) if len(y01) else 0.5
        self.maps_ = {}

        X_encoded = X_obj.copy()
        for col in self.cat_features:
            values = X_obj[:, col]
            mapping = {}
            if self.ordered:
                counts = {}
                sums = {}
                encoded_col = np.empty(values.shape[0], dtype=float)
                for i, val in enumerate(values):
                    cnt = counts.get(val, 0.0)
                    sm = sums.get(val, 0.0)
                    encoded_col[i] = (sm + self.prior_weight * self.global_mean_) / (cnt + self.prior_weight)
                    counts[val] = cnt + 1.0
                    sums[val] = sm + y01[i]
                for val in counts:
                    mapping[str(val)] = (sums[val] + self.prior_weight * self.global_mean_) / (counts[val] + self.prior_weight)
                X_encoded[:, col] = encoded_col
            else:
                unique_values, inverse = np.unique(values.astype(str), return_inverse=True)
                sums = np.bincount(inverse, weights=y01, minlength=len(unique_values))
                counts = np.bincount(inverse, minlength=len(unique_values))
                means = (sums + self.prior_weight * self.global_mean_) / (counts + self.prior_weight)
                mapping = {val: float(mean) for val, mean in zip(unique_values, means)}
                X_encoded[:, col] = means[inverse]
            self.maps_[col] = mapping

        self.train_encoded_ = self._safe_float(X_encoded)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_obj = np.asarray(X, dtype=object).copy()
        for col in self.cat_features:
            mapping = self.maps_.get(col, {})
            X_obj[:, col] = np.array([mapping.get(str(v), self.global_mean_) for v in X_obj[:, col]], dtype=float)
        return self._safe_float(X_obj)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        if self.ordered and self.train_encoded_ is not None:
            return self.train_encoded_.copy()
        return self.transform(X)

    @staticmethod
    def _safe_float(X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=float)
        for j in range(X.shape[1]):
            col = X[:, j]
            try:
                out[:, j] = col.astype(float)
            except Exception:
                vals, inv = np.unique(col.astype(str), return_inverse=True)
                out[:, j] = inv.astype(float)
        return out


class Quantizer:

    def __init__(self, quantization_type: str | None = None, nbins: int = 255, random_state: int | None = None):
        self.quantization_type = quantization_type
        self.nbins = int(nbins)
        self.random_state = random_state
        self.borders_: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        self.borders_ = []
        if self.quantization_type is None:
            return self
        qtype = self.quantization_type.lower()
        max_borders = max(1, self.nbins - 1)
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[np.isfinite(col)]
            if col.size <= 1 or np.nanmin(col) == np.nanmax(col):
                self.borders_.append(np.array([], dtype=float))
                continue
            if qtype == "uniform":
                borders = np.linspace(col.min(), col.max(), self.nbins + 1)[1:-1]
            elif qtype == "quantile":
                qs = np.linspace(0, 1, self.nbins + 1)[1:-1]
                borders = np.quantile(col, qs)
            elif qtype in {"min_entropy", "piecewise"} and y is not None:
                tree = DecisionTreeClassifier(
                    max_leaf_nodes=min(self.nbins, max(2, np.unique(col).size)),
                    min_samples_leaf=max(2, int(0.01 * len(col))),
                    random_state=self.random_state,
                )
                finite_mask = np.isfinite(X[:, j])
                tree.fit(X[finite_mask][:, [j]], (np.asarray(y)[finite_mask] == 1).astype(int))
                borders = tree.tree_.threshold[tree.tree_.threshold != -2]
            else:
                qs = np.linspace(0, 1, self.nbins + 1)[1:-1]
                borders = np.quantile(col, qs)
            borders = np.unique(np.asarray(borders, dtype=float))[:max_borders]
            self.borders_.append(borders)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.quantization_type is None:
            return X
        out = np.empty(X.shape, dtype=float)
        for j, borders in enumerate(self.borders_):
            col = X[:, j]
            codes = np.searchsorted(borders, col, side="right").astype(float)
            codes[~np.isfinite(col)] = -1.0
            out[:, j] = codes
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class Boosting(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = False,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        cat_features: Iterable[int] | None = None,
        ordered_cat_encoding: bool = False,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = "Bernoulli",
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        quantization_type: str | None = None,
        nbins: int = 255,
        dart: bool = False,
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        if not (0 < learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0 < subsample <= 1):
            raise ValueError("subsample must be in (0, 1]")
        if not (0 < rsm <= 1):
            raise ValueError("rsm must be in (0, 1]")

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else dict(base_model_params)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.verbose = verbose

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.cat_features = None if cat_features is None else list(cat_features)
        self.ordered_cat_encoding = ordered_cat_encoding
        self.subsample = float(subsample)
        self.bagging_temperature = float(bagging_temperature)
        self.bootstrap_type = bootstrap_type
        self.rsm = float(rsm)
        self.goss = bool(goss)
        self.goss_k = float(goss_k)
        self.quantization_type = quantization_type
        self.nbins = int(nbins)
        self.dart = bool(dart)
        self.dropout_rate = float(dropout_rate)

        self.models: list = []
        self.gammas: list[float] = []
        self.feature_indices_: list[np.ndarray] = []
        self.history = defaultdict(list)
        self.classes_ = np.array([-1, 1])
        self.rng_ = np.random.default_rng(self.random_state)
        self.encoder_: TargetEncoder | None = None
        self.quantizer_: Quantizer | None = None
        self.n_features_in_: int | None = None

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def loss_fn(self, y: np.ndarray, z: np.ndarray) -> float:
        yz = np.asarray(y) * np.asarray(z)
        return float(np.logaddexp(0.0, -yz).mean())

    def loss_derivative(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return -np.asarray(y) * self.sigmoid(-np.asarray(y) * np.asarray(z))

    def _negative_gradient(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return -self.loss_derivative(y, z)

    def _preprocess_fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_proc = np.asarray(X, dtype=object)
        self.encoder_ = TargetEncoder(self.cat_features, ordered=self.ordered_cat_encoding)
        X_proc = self.encoder_.fit_transform(X_proc, y)
        self.quantizer_ = Quantizer(self.quantization_type, self.nbins, self.random_state)
        X_proc = self.quantizer_.fit_transform(X_proc, y)
        self.n_features_in_ = X_proc.shape[1]
        return X_proc

    def _preprocess_transform(self, X: np.ndarray) -> np.ndarray:
        X_proc = np.asarray(X, dtype=object)
        if self.encoder_ is not None:
            X_proc = self.encoder_.transform(X_proc)
        X_proc = np.asarray(X_proc, dtype=float)
        if self.quantizer_ is not None:
            X_proc = self.quantizer_.transform(X_proc)
        return X_proc

    def _make_model(self):
        params = dict(self.base_model_params)
        if "random_state" not in params:
            try:
                import inspect
                if "random_state" in inspect.signature(self.base_model_class).parameters:
                    params["random_state"] = int(self.rng_.integers(0, 2**31 - 1))
            except (TypeError, ValueError):
                pass
        return self.base_model_class(**params)

    def _choose_features(self, n_features: int) -> np.ndarray:
        m = max(1, int(np.ceil(self.rsm * n_features)))
        if m >= n_features:
            return np.arange(n_features)
        return np.sort(self.rng_.choice(n_features, size=m, replace=False))

    def _sample_rows(self, anti_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n = anti_grad.shape[0]
        indices = np.arange(n)
        weights = None

        if self.goss:
            k = max(1, int(np.ceil(self.goss_k * n)))
            big = np.argpartition(np.abs(anti_grad), -k)[-k:]
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[big] = False
            rest = indices[rest_mask]
            small = rest[self.rng_.random(rest.shape[0]) < self.subsample]
            if small.size == 0 and rest.size > 0:
                small = self.rng_.choice(rest, size=1, replace=False)
            chosen = np.concatenate([big, small])
            weights = np.ones(chosen.shape[0])
            if small.size > 0:
                weights[len(big):] = (1.0 - self.goss_k) / max(self.subsample, 1e-12)
            return chosen, weights

        if self.bootstrap_type is None or self.subsample >= 1.0 and str(self.bootstrap_type).lower() == "bernoulli":
            return indices, None

        btype = str(self.bootstrap_type).lower()
        if btype == "bernoulli":
            mask = self.rng_.random(n) < self.subsample
            if not mask.any():
                mask[self.rng_.integers(0, n)] = True
            return indices[mask], None
        if btype == "bayesian":
            u = np.clip(self.rng_.random(n), 1e-12, 1.0)
            weights = (-np.log(u)) ** self.bagging_temperature
            return indices, weights
        return indices, None

    def _predict_raw_preprocessed(self, X: np.ndarray, model_indices: Iterable[int] | None = None) -> np.ndarray:
        if model_indices is None:
            model_indices = range(len(self.models))
        pred = np.zeros(X.shape[0], dtype=float)
        for i in model_indices:
            cols = self.feature_indices_[i]
            pred += self.learning_rate * self.gammas[i] * self.models[i].predict(X[:, cols])
        return pred

    def partial_fit(self, X: np.ndarray, y: np.ndarray, current_predictions: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if current_predictions is None:
            current_predictions = self._predict_raw_preprocessed(X)

        gradient_predictions = current_predictions
        dropped = []
        if self.dart and len(self.models) > 0:
            n_drop = max(1, int(np.ceil(self.dropout_rate * len(self.models))))
            dropped = list(self.rng_.choice(len(self.models), size=n_drop, replace=False))
            kept = [i for i in range(len(self.models)) if i not in set(dropped)]
            gradient_predictions = self._predict_raw_preprocessed(X, kept)

        anti_grad = self._negative_gradient(y, gradient_predictions)
        row_idx, sample_weight = self._sample_rows(anti_grad)
        feature_idx = self._choose_features(X.shape[1])

        model = self._make_model()
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        try:
            model.fit(X[row_idx][:, feature_idx], anti_grad[row_idx], **fit_kwargs)
        except TypeError:
            model.fit(X[row_idx][:, feature_idx], anti_grad[row_idx])

        new_predictions = model.predict(X[:, feature_idx])
        gamma = self.find_optimal_gamma(y, current_predictions, self.learning_rate * new_predictions)

        if self.dart and dropped:
            k = len(dropped)
            for i in dropped:
                self.gammas[i] *= k / (k + 1.0)
            gamma = gamma / (k + 1.0)

        self.models.append(model)
        self.gammas.append(float(gamma))
        self.feature_indices_.append(feature_idx)
        return self.learning_rate * gamma * new_predictions

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Tuple[np.ndarray, np.ndarray] | None = None,
        use_best_model: bool = False,
    ):
        self.models = []
        self.gammas = []
        self.feature_indices_ = []
        self.history = defaultdict(list)
        self.rng_ = np.random.default_rng(self.random_state)

        X_train_proc = self._preprocess_fit(X_train, y_train)
        y_train = np.asarray(y_train)
        train_predictions = np.zeros(X_train_proc.shape[0], dtype=float)

        if eval_set is not None:
            X_valid, y_valid = eval_set
            X_valid_proc = self._preprocess_transform(X_valid)
            y_valid = np.asarray(y_valid)
            valid_predictions = np.zeros(X_valid_proc.shape[0], dtype=float)
        else:
            X_valid_proc = y_valid = valid_predictions = None

        best_iter = -1
        best_value = None
        rounds_without_improvement = 0
        metric_for_es = self.eval_metric
        if metric_for_es is None and eval_set is not None:
            metric_for_es = "valid_loss"

        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm(iterator)

        for _ in iterator:
            update = self.partial_fit(X_train_proc, y_train, train_predictions)
            train_predictions += update
            self._append_history("train", y_train, train_predictions)

            if eval_set is not None:
                last_model = self.models[-1]
                last_gamma = self.gammas[-1]
                last_cols = self.feature_indices_[-1]
                valid_predictions += self.learning_rate * last_gamma * last_model.predict(X_valid_proc[:, last_cols])
                self._append_history("valid", y_valid, valid_predictions)

                if metric_for_es in self.history:
                    value = self.history[metric_for_es][-1]
                    maximize = "auc" in metric_for_es or "roc" in metric_for_es
                    improved = (
                        best_value is None
                        or (maximize and value > best_value + 1e-12)
                        or ((not maximize) and value < best_value - 1e-12)
                    )
                    if improved:
                        best_value = value
                        best_iter = len(self.models) - 1
                        rounds_without_improvement = 0
                    else:
                        rounds_without_improvement += 1

                    if self.early_stopping_rounds and rounds_without_improvement >= self.early_stopping_rounds:
                        break

        if use_best_model and best_iter >= 0:
            keep = best_iter + 1
            self.models = self.models[:keep]
            self.gammas = self.gammas[:keep]
            self.feature_indices_ = self.feature_indices_[:keep]
            for key in list(self.history.keys()):
                self.history[key] = self.history[key][:keep]

        for key in list(self.history.keys()):
            self.history[key] = np.asarray(self.history[key])
        return self

    def _append_history(self, prefix: str, y: np.ndarray, raw_pred: np.ndarray):
        self.history[f"{prefix}_loss"].append(self.loss_fn(y, raw_pred))
        try:
            self.history[f"{prefix}_roc_auc"].append(roc_auc_score(y == 1, self.sigmoid(raw_pred)))
        except ValueError:
            self.history[f"{prefix}_roc_auc"].append(np.nan)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_proc = self._preprocess_transform(X)
        return self._predict_raw_preprocessed(X_proc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.decision_function(X)
        p_pos = self.sigmoid(raw)
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, 1, -1)

    def find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        gammas = np.linspace(start=0.0, stop=1.0, num=101)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return float(gammas[int(np.argmin(losses))])

    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(np.asarray(y) == 1, self.predict_proba(X)[:, 1])

    def plot_history(self, keys: str | Iterable[str]):
        if isinstance(keys, str):
            keys = [keys]
        plt.figure(figsize=(8, 5))
        for key in keys:
            if key not in self.history:
                raise KeyError(f"Unknown history key: {key}")
            plt.plot(self.history[key], label=key)
        plt.xlabel("iteration")
        plt.ylabel("metric")
        plt.legend()
        plt.grid(True)
        plt.show()

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.models or self.n_features_in_ is None:
            return np.array([])
        importances = np.zeros(self.n_features_in_, dtype=float)
        for model, gamma, cols in zip(self.models, self.gammas, self.feature_indices_):
            if hasattr(model, "feature_importances_"):
                gain = np.asarray(model.feature_importances_, dtype=float)
                importances[cols] += abs(gamma) * gain
        s = importances.sum()
        if s > 0:
            importances /= s
        return importances
