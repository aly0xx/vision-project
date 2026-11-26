# Step 5: trains KNN, SVM and a small neural network on synthetic digits, each with and without PCA.
# Settings are tuned on the validation set only, then every model is tested once on the real photo digits.

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from dataset import load_real, load_synthetic
from utils import BASE_DIR, RESULTS_DIR


MODELS_PATH = BASE_DIR / "models" / "classifiers.pkl"
LABELS = list(range(1, 10))


def choose_pca(X_train, variance=0.95):
	pca = PCA().fit(X_train)
	n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), variance)) + 1
	return n, pca.explained_variance_ratio_


def make_model(kind, params, n_pca):
	if kind == "knn":
		model = KNeighborsClassifier(n_neighbors=params)
	else:
		model = SVC(kernel=params[0], C=params[1], gamma="scale", probability=True, random_state=0)
	if n_pca:
		return make_pipeline(PCA(n_components=n_pca, random_state=0), model)
	return model


def tune(kind, options, n_pca, data):
	X_train, X_val, y_train, y_val = data
	best, best_acc, best_params = None, -1, None
	for params in options:
		model = make_model(kind, params, n_pca).fit(X_train, y_train)
		acc = accuracy_score(y_val, model.predict(X_val))
		print(f"  {kind} {params}: val acc {acc:.3f}")
		if acc > best_acc:
			best, best_acc, best_params = model, acc, params
	return best, best_acc, best_params


def train_mlp(hidden, data, n_pca, epochs=40):
	X_train, X_val, y_train, y_val = data
	pca = None
	if n_pca:
		pca = PCA(n_components=n_pca, random_state=0).fit(X_train)
		X_train, X_val = pca.transform(X_train), pca.transform(X_val)

	mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", learning_rate_init=1e-3,
						batch_size=64, random_state=0)
	rng = np.random.default_rng(0)
	train_loss, val_loss = [], []
	best, best_loss = None, np.inf
	for epoch in range(epochs):
		order = rng.permutation(len(X_train))
		for start in range(0, len(order), 64):
			batch = order[start:start + 64]
			mlp.partial_fit(X_train[batch], y_train[batch], classes=LABELS)
		train_loss.append(log_loss(y_train, mlp.predict_proba(X_train), labels=LABELS))
		val_loss.append(log_loss(y_val, mlp.predict_proba(X_val), labels=LABELS))
		if val_loss[-1] < best_loss:
			best, best_loss = copy.deepcopy(mlp), val_loss[-1]

	model = make_pipeline(pca, best) if pca is not None else best
	acc = accuracy_score(y_val, best.predict(X_val))
	return model, acc, train_loss, val_loss


def tune_mlp(options, n_pca, data):
	best = None
	for hidden in options:
		model, acc, train_loss, val_loss = train_mlp(hidden, data, n_pca)
		print(f"  mlp {hidden}: val acc {acc:.3f}, best epoch {int(np.argmin(val_loss)) + 1}")
		if best is None or acc > best[1]:
			best = (model, acc, hidden, train_loss, val_loss)
	return best


def train_all():
	data = load_synthetic()
	n_pca, ratios = choose_pca(data[0])
	print(f"PCA keeps {n_pca} components for 95% of the variance")

	results = {}
	curves = {}
	for use_pca in [False, True]:
		n = n_pca if use_pca else 0
		suffix = " + PCA" if use_pca else ""

		model, acc, params = tune("knn", [1, 3, 5, 7, 9], n, data)
		results["KNN" + suffix] = (model, acc, f"k={params}")

		model, acc, params = tune("svm", [("linear", 1), ("rbf", 1), ("rbf", 10)], n, data)
		results["SVM" + suffix] = (model, acc, f"{params[0]}, C={params[1]}")

		model, acc, hidden, train_loss, val_loss = tune_mlp([(64,), (128,), (128, 64)], n, data)
		results["MLP" + suffix] = (model, acc, f"hidden={hidden}")
		curves["MLP" + suffix] = (train_loss, val_loss)

	return results, curves, n_pca, ratios


def plot_curves(curves, ratios, n_pca):
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	for ax, (name, (train_loss, val_loss)) in zip(axes, curves.items()):
		ax.plot(range(1, len(train_loss) + 1), train_loss, label="train")
		ax.plot(range(1, len(val_loss) + 1), val_loss, label="validation")
		ax.set_title(f"{name} loss")
		ax.set_xlabel("epoch")
		ax.set_ylabel("cross-entropy")
		ax.legend()
	axes[2].plot(np.cumsum(ratios)[:200])
	axes[2].axvline(n_pca, color="gray", linestyle="--")
	axes[2].set_title(f"PCA explained variance ({n_pca} components = 95%)")
	axes[2].set_xlabel("components")
	fig.tight_layout()
	fig.savefig(RESULTS_DIR / "step5_training.png", dpi=120)
	plt.close(fig)


def test_all(results):
	X_test, y_test, _ = load_real()
	fig, axes = plt.subplots(2, 3, figsize=(15, 9))
	report = []
	rows = []
	for ax, (name, (model, val_acc, params)) in zip(axes.ravel(), results.items()):
		pred = model.predict(X_test)
		acc = accuracy_score(y_test, pred)
		rows.append((name, params, val_acc, acc))
		cm = confusion_matrix(y_test, pred, labels=LABELS)
		ax.imshow(cm, cmap="Blues")
		for i in range(9):
			for j in range(9):
				if cm[i, j]:
					ax.text(j, i, cm[i, j], ha="center", va="center",
							color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
		ax.set_xticks(range(9), LABELS)
		ax.set_yticks(range(9), LABELS)
		ax.set_xlabel("predicted")
		ax.set_ylabel("true")
		ax.set_title(f"{name}: test acc {acc:.3f}")
		report.append(f"== {name} ({params}) ==\n" +
					  classification_report(y_test, pred, labels=LABELS, digits=3, zero_division=0))
	fig.tight_layout()
	fig.savefig(RESULTS_DIR / "step5_confusion.png", dpi=120)
	plt.close(fig)
	(RESULTS_DIR / "step5_report.txt").write_text("\n".join(report))
	return rows, len(y_test)


if __name__ == "__main__":
	RESULTS_DIR.mkdir(exist_ok=True)
	results, curves, n_pca, ratios = train_all()
	plot_curves(curves, ratios, n_pca)

	best_name = max(results, key=lambda name: results[name][1])
	MODELS_PATH.parent.mkdir(exist_ok=True)
	with open(MODELS_PATH, "wb") as f:
		pickle.dump({"models": {k: v[0] for k, v in results.items()}, "best": best_name}, f)

	rows, n_test = test_all(results)
	print(f"\n{'model':12s} {'settings':22s} {'val acc':>8s} {'test acc':>9s}   ({n_test} real test digits)")
	for name, params, val_acc, test_acc in rows:
		print(f"{name:12s} {params:22s} {val_acc:8.3f} {test_acc:9.3f}")
	print(f"\nbest on validation: {best_name} (used by the reader)")
