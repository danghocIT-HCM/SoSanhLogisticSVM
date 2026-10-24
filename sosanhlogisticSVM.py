import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
    RepeatedStratifiedKFold, cross_validate, learning_curve,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss, classification_report,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CẤU HÌNH CHUNG
# ----------------------------------------------------------------------------
SEED = 42
OUT_DIR = "ket_qua"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"Logistic Regression": "#2a7fff", "SVM": "#ff7a2a"}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  [+] Đã lưu: {path}")


def banner(text):
    print(f"\n--- {text} ---")


# ----------------------------------------------------------------------------
# 1. NẠP DỮ LIỆU & KHÁM PHÁ
# ----------------------------------------------------------------------------
banner("1. KHÁM PHÁ DỮ LIỆU")
data = load_breast_cancer()
X, y = data.data, data.target    # 0 = ác tính (malignant), 1 = lành tính (benign)
feature_names = data.feature_names
class_names = ["Ác tính (malignant)", "Lành tính (benign)"]

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
print(f"Số mẫu: {X.shape[0]} | Số đặc trưng: {X.shape[1]}")
print(f"Phân bố lớp: ác tính = {(y == 0).sum()} ({(y == 0).mean():.1%}), "
      f"lành tính = {(y == 1).sum()} ({(y == 1).mean():.1%})")
print(f"Giá trị thiếu: {df.isna().sum().sum()}")

# --- Biểu đồ EDA ---
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].bar(class_names, [(y == 0).sum(), (y == 1).sum()],
            color=["#d64545", "#3aa655"])
for i, v in enumerate([(y == 0).sum(), (y == 1).sum()]):
    axes[0].text(i, v + 4, str(v), ha="center", fontweight="bold")
axes[0].set_title("Phân bố lớp")
axes[0].set_ylabel("Số mẫu")

# Top 10 đặc trưng tương quan mạnh nhất với nhãn
corr = df.corr()["target"].drop("target").sort_values(key=np.abs, ascending=False)
top = corr.head(10).iloc[::-1]
axes[1].barh(top.index, top.values,
             color=["#3aa655" if v > 0 else "#d64545" for v in top.values])
axes[1].set_title("Top 10 đặc trưng tương quan với nhãn")
axes[1].set_xlabel("Hệ số tương quan Pearson")

# Ma trận tương quan của 10 đặc trưng "mean"
mean_cols = [c for c in feature_names if c.startswith("mean")]
cm_corr = df[mean_cols].corr()
im = axes[2].imshow(cm_corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[2].set_xticks(range(len(mean_cols)))
axes[2].set_yticks(range(len(mean_cols)))
axes[2].set_xticklabels([c.replace("mean ", "") for c in mean_cols],
                        rotation=75, ha="right", fontsize=8)
axes[2].set_yticklabels([c.replace("mean ", "") for c in mean_cols], fontsize=8)
axes[2].set_title("Tương quan giữa các đặc trưng 'mean'")
axes[2].grid(False)
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.suptitle("Khám phá dữ liệu Breast Cancer Wisconsin", fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "01_eda.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. CHIA DỮ LIỆU & TINH CHỈNH SIÊU THAM SỐ
# ----------------------------------------------------------------------------
banner("2. CHIA DỮ LIỆU & TINH CHỈNH SIÊU THAM SỐ")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)
print(f"Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu (chia theo tỉ lệ lớp)")

# Dùng Pipeline để StandardScaler chỉ fit trên train của từng fold -> không rò rỉ dữ liệu
pipes = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=SEED)),
    ]),
}

param_grids = {
    "Logistic Regression": {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    "SVM": [
        {"clf__kernel": ["linear"], "clf__C": np.logspace(-3, 2, 6)},
        {"clf__kernel": ["rbf"], "clf__C": np.logspace(-2, 3, 6),
         "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
    ],
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
best_models, grid_results = {}, {}
for name, pipe in pipes.items():
    t0 = time.perf_counter()
    gs = GridSearchCV(pipe, param_grids[name], cv=cv5, scoring="roc_auc",
                      n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    dt = time.perf_counter() - t0
    best_models[name] = gs.best_estimator_
    grid_results[name] = gs
    clean = {k.replace("clf__", ""): (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in gs.best_params_.items()}
    print(f"\n{name}\n  Tham số tốt nhất : {clean}\n"
          f"  ROC-AUC (CV 5-fold): {gs.best_score_:.4f}   (tìm kiếm mất {dt:.1f}s)")

# --- Biểu đồ ảnh hưởng của C ---
fig, ax = plt.subplots(figsize=(8, 4.6))
res = pd.DataFrame(grid_results["Logistic Regression"].cv_results_)
ax.semilogx(res["param_clf__C"].astype(float), res["mean_test_score"],
            "o-", color=COLORS["Logistic Regression"], label="Logistic Regression")
res = pd.DataFrame(grid_results["SVM"].cv_results_)
for kernel, ls in [("linear", "s--"), ("rbf", "^-")]:
    sub = res[res["param_clf__kernel"] == kernel]
    best_per_c = sub.groupby(sub["param_clf__C"].astype(float))["mean_test_score"].max()
    ax.semilogx(best_per_c.index, best_per_c.values, ls,
                color=COLORS["SVM"], alpha=1 if kernel == "rbf" else 0.55,
                label=f"SVM ({kernel})")
ax.set_xlabel("Tham số điều chuẩn C (thang log)")
ax.set_ylabel("ROC-AUC (CV 5-fold)")
ax.set_title("Độ nhạy của mô hình với siêu tham số C", fontweight="bold")
ax.legend()
fig.tight_layout()
save(fig, "02_anh_huong_C.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. ĐÁNH GIÁ TRÊN TẬP TEST
# ----------------------------------------------------------------------------
banner("3. ĐÁNH GIÁ TRÊN TẬP TEST")
results, preds, probas, scores = {}, {}, {}, {}
for name, model in best_models.items():
    # Đo thời gian huấn luyện lại (với tham số tốt nhất) và thời gian dự đoán
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        y_pred = model.predict(X_test)
    predict_time = (time.perf_counter() - t0) / 20

    y_prob = model.predict_proba(X_test)[:, 1]
    preds[name], probas[name] = y_pred, y_prob

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall (Sensitivity)": recall_score(y_test, y_pred),
        "Specificity": tn / (tn + fp),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Avg Precision (PR-AUC)": average_precision_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Brier score (↓ tốt)": brier_score_loss(y_test, y_prob),
        "Bỏ sót ung thư (FP*)": fp,   # dự đoán lành tính nhưng thực tế ác tính -> nguy hiểm
        "Báo động nhầm (FN*)": fn,
        "Thời gian train (ms)": train_time * 1000,
        "Thời gian predict (ms)": predict_time * 1000,
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("* Lưu ý: nhãn 1 = lành tính (positive). FP = ca ÁC TÍNH bị dự đoán thành lành tính "
      "(bỏ sót bệnh – lỗi nghiêm trọng nhất trong y tế).")

res_df = pd.DataFrame(results).T
pd.set_option("display.width", 200)
print("\nBẢNG TỔNG HỢP (tập test):")
print(res_df.round(4).T.to_string())
res_df.round(4).to_csv(os.path.join(OUT_DIR, "bang_so_sanh_test.csv"),
                        encoding="utf-8-sig")

# --- Biểu đồ cột so sánh chỉ số ---
metrics_plot = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity",
                "F1-score", "ROC-AUC", "MCC"]
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(metrics_plot))
w = 0.38
for i, name in enumerate(best_models):
    vals = [res_df.loc[name, m] for m in metrics_plot]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=COLORS[name])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot, rotation=15)
ax.set_ylim(min(0.85, res_df[metrics_plot].values.min() - 0.03), 1.03)
ax.set_title("So sánh các chỉ số trên tập test", fontweight="bold")
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "03_so_sanh_chi_so.png")
plt.close(fig)

# --- Radar chart ---
radar_metrics = ["Accuracy", "Precision", "Recall (Sensitivity)",
                 "Specificity", "F1-score", "ROC-AUC"]
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]
fig = plt.figure(figsize=(6.5, 6.5))
ax = plt.subplot(111, polar=True)
for name in best_models:
    vals = [res_df.loc[name, m] for m in radar_metrics]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", color=COLORS[name], label=name, linewidth=2)
    ax.fill(angles, vals, color=COLORS[name], alpha=0.12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
lo = max(0.8, res_df[radar_metrics].values.min() - 0.03)
ax.set_ylim(lo, 1.0)
ax.set_title("Radar chart hiệu năng", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
fig.tight_layout()
save(fig, "04_radar.png")
plt.close(fig)

# --- Confusion matrix ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, name in zip(axes, best_models):
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds[name], display_labels=["Ác tính", "Lành tính"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name, color=COLORS[name], fontweight="bold")
    ax.grid(False)
fig.suptitle("Ma trận nhầm lẫn (tập test)", fontweight="bold")
fig.tight_layout()
save(fig, "05_confusion_matrix.png")
plt.close(fig)

# --- ROC & Precision-Recall ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name in best_models:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    axes[0].plot(fpr, tpr, lw=2, color=COLORS[name],
                 label=f"{name} (AUC = {res_df.loc[name, 'ROC-AUC']:.4f})")
    p, r, _ = precision_recall_curve(y_test, probas[name])
    axes[1].plot(r, p, lw=2, color=COLORS[name],
                 label=f"{name} (AP = {res_df.loc[name, 'Avg Precision (PR-AUC)']:.4f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ngẫu nhiên")
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="Đường cong ROC")
axes[0].legend(loc="lower right")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Đường cong Precision-Recall")
axes[1].legend(loc="lower left")
axes[1].set_ylim(0.85, 1.01)
fig.tight_layout()
save(fig, "06_roc_pr.png")
plt.close(fig)

# --- Calibration ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Hiệu chuẩn hoàn hảo")
for name in best_models:
    frac_pos, mean_pred = calibration_curve(y_test, probas[name], n_bins=8,
                                            strategy="quantile")
    axes[0].plot(mean_pred, frac_pos, "o-", color=COLORS[name], label=name)
    axes[1].hist(probas[name], bins=20, alpha=0.55, color=COLORS[name], label=name)
axes[0].set(xlabel="Xác suất dự đoán trung bình", ylabel="Tỉ lệ thực tế dương",
            title="Đường cong hiệu chuẩn (calibration)")
axes[0].legend()
axes[1].set(xlabel="Xác suất dự đoán 'lành tính'", ylabel="Số mẫu",
            title="Phân phối xác suất dự đoán")
axes[1].legend()
fig.tight_layout()
save(fig, "07_calibration.png")
plt.close(fig)

# --- Tác động của ngưỡng quyết định ---
thresholds = np.linspace(0.05, 0.95, 91)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, name in zip(axes, best_models):
    prec = [precision_score(y_test, probas[name] >= t, zero_division=0) for t in thresholds]
    rec = [recall_score(y_test, probas[name] >= t) for t in thresholds]
    f1s = [f1_score(y_test, probas[name] >= t) for t in thresholds]
    ax.plot(thresholds, prec, label="Precision")
    ax.plot(thresholds, rec, label="Recall")
    ax.plot(thresholds, f1s, "k--", label="F1")
    ax.axvline(0.5, color="gray", ls=":", alpha=0.7)
    ax.set_title(name, color=COLORS[name], fontweight="bold")
    ax.set_xlabel("Ngưỡng quyết định")
    ax.legend(loc="lower center")
axes[0].set_ylabel("Giá trị")
fig.suptitle("Ảnh hưởng của ngưỡng quyết định đến Precision / Recall / F1",
             fontweight="bold")
fig.tight_layout()
save(fig, "08_nguong_quyet_dinh.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. CROSS-VALIDATION LẶP LẠI & KIỂM ĐỊNH THỐNG KÊ
# ----------------------------------------------------------------------------
banner("4. CROSS-VALIDATION LẶP LẠI (5-fold x 10 lần) & KIỂM ĐỊNH THỐNG KÊ")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_scores = {}
for name, model in best_models.items():
    cv_scores[name] = cross_validate(model, X, y, cv=rcv, scoring=scoring, n_jobs=-1)

rows = []
for name in best_models:
    row = {"Mô hình": name}
    for s in scoring:
        arr = cv_scores[name][f"test_{s}"]
        row[s] = f"{arr.mean():.4f} ± {arr.std():.4f}"
    rows.append(row)
cv_table = pd.DataFrame(rows).set_index("Mô hình")
print(cv_table.to_string())
cv_table.to_csv(os.path.join(OUT_DIR, "bang_cross_validation.csv"), encoding="utf-8-sig")

print("\nKiểm định t ghép cặp (cùng các fold) – H0: hai mô hình có hiệu năng như nhau")
sig_info = {}
for s in scoring:
    a = cv_scores["Logistic Regression"][f"test_{s}"]
    b = cv_scores["SVM"][f"test_{s}"]
    t_stat, p = stats.ttest_rel(a, b)
    sig_info[s] = p
    kl = "KHÁC BIỆT có ý nghĩa" if p < 0.05 else "không khác biệt có ý nghĩa"
    print(f"  {s:<10} : chênh lệch TB = {(a - b).mean():+.4f} | p = {p:.4f} -> {kl} (α=0.05)")
print("  (Lưu ý: các fold lặp lại không hoàn toàn độc lập nên p-value có xu hướng "
      "hơi lạc quan; hãy xem đây là chỉ dẫn tham khảo.)")

# Boxplot kết quả CV
fig, axes = plt.subplots(1, len(scoring), figsize=(16, 4.6))
for ax, s in zip(axes, scoring):
    bp = ax.boxplot([cv_scores[n][f"test_{s}"] for n in best_models],
                    labels=["Logistic", "SVM"], patch_artist=True, widths=0.55)
    for patch, n in zip(bp["boxes"], best_models):
        patch.set_facecolor(COLORS[n])
        patch.set_alpha(0.6)
    ax.set_title(f"{s}\n(p = {sig_info[s]:.3f})")
fig.suptitle("Phân phối điểm Cross-Validation (5-fold × 10 lần lặp)",
             fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "09_cv_boxplot.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. LEARNING CURVE
# ----------------------------------------------------------------------------
banner("5. LEARNING CURVE")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, name in zip(axes, best_models):
    sizes, tr, va = learning_curve(
        best_models[name], X, y, cv=cv5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=SEED)
    ax.plot(sizes, tr.mean(1), "o-", color="#555", label="Train")
    ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1),
                    color="#555", alpha=0.12)
    ax.plot(sizes, va.mean(1), "o-", color=COLORS[name], label="Validation")
    ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1),
                    color=COLORS[name], alpha=0.18)
    ax.set_title(name, color=COLORS[name], fontweight="bold")
    ax.set_xlabel("Số mẫu huấn luyện")
    ax.legend(loc="lower right")
axes[0].set_ylabel("Accuracy")
fig.suptitle("Learning curve – chẩn đoán overfitting / underfitting", fontweight="bold")
fig.tight_layout()
save(fig, "10_learning_curve.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. RANH GIỚI QUYẾT ĐỊNH TRÊN KHÔNG GIAN PCA 2D
# ----------------------------------------------------------------------------
banner("6. RANH GIỚI QUYẾT ĐỊNH (PCA 2D)")
scaler2 = StandardScaler().fit(X_train)
pca = PCA(n_components=2, random_state=SEED).fit(scaler2.transform(X_train))
Z_train = pca.transform(scaler2.transform(X_train))
Z_test = pca.transform(scaler2.transform(X_test))
print(f"2 thành phần PCA giữ lại {pca.explained_variance_ratio_.sum():.1%} phương sai")

# Dùng đúng siêu tham số tốt nhất, huấn luyện lại trên không gian 2D chỉ để TRỰC QUAN HÓA
def clf_params(model):
    return model.named_steps["clf"].get_params()

lr2 = LogisticRegression(C=clf_params(best_models["Logistic Regression"])["C"],
                         max_iter=5000).fit(Z_train, y_train)
svm_p = clf_params(best_models["SVM"])
svm2 = SVC(kernel=svm_p["kernel"], C=svm_p["C"], gamma=svm_p["gamma"]).fit(Z_train, y_train)

pad = 1.0
xx, yy = np.meshgrid(
    np.linspace(Z_train[:, 0].min() - pad, Z_train[:, 0].max() + pad, 400),
    np.linspace(Z_train[:, 1].min() - pad, Z_train[:, 1].max() + pad, 400))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
for ax, (name, clf) in zip(axes, [("Logistic Regression", lr2), ("SVM", svm2)]):
    zz = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.18, cmap="coolwarm")
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2)
    for cls, c, lab in [(0, "#d64545", "Ác tính"), (1, "#3aa655", "Lành tính")]:
        m = y_test == cls
        ax.scatter(Z_test[m, 0], Z_test[m, 1], c=c, s=26, edgecolor="k",
                   linewidth=0.4, label=lab)
    acc2 = accuracy_score(y_test, clf.predict(Z_test))
    ax.set_title(f"{name}  (acc trên 2D = {acc2:.3f})", color=COLORS[name],
                 fontweight="bold")
    ax.set_xlabel("PC1")
    ax.grid(False)
    if name == "SVM" and svm_p["kernel"] != "linear":
        sv = Z_train[svm2.support_]
        ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolors="none", edgecolors="gold",
                   linewidths=0.6, alpha=0.6, label="Support vectors")
    ax.legend(loc="best", fontsize=8)
axes[0].set_ylabel("PC2")
fig.suptitle("Ranh giới quyết định (điểm là dữ liệu TEST, chiếu xuống PCA 2D)\n"
             "Mô hình thực tế dùng đủ 30 đặc trưng – hình này chỉ để minh họa hình dạng ranh giới",
             fontweight="bold")
fig.tight_layout()
save(fig, "11_decision_boundary.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. ĐỘ QUAN TRỌNG ĐẶC TRƯNG (Logistic) & TÍNH ỔN ĐỊNH CỦA SVM
# ----------------------------------------------------------------------------
banner("7. ĐỘ QUAN TRỌNG ĐẶC TRƯNG")
lr_coef = best_models["Logistic Regression"].named_steps["clf"].coef_[0]
order = np.argsort(np.abs(lr_coef))[::-1][:12][::-1]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.barh(feature_names[order], lr_coef[order],
        color=["#3aa655" if v > 0 else "#d64545" for v in lr_coef[order]])
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Logistic Regression – 12 hệ số lớn nhất\n"
             "(xanh: đẩy về 'lành tính' | đỏ: đẩy về 'ác tính')", fontweight="bold")
ax.set_xlabel("Hệ số (đặc trưng đã chuẩn hóa)")
fig.tight_layout()
save(fig, "12_feature_importance_logistic.png")
plt.close(fig)

svm_clf = best_models["SVM"].named_steps["clf"]
print(f"SVM kernel '{svm_clf.kernel}': dùng {len(svm_clf.support_)} support vectors "
      f"trên {len(X_train)} mẫu huấn luyện ({len(svm_clf.support_) / len(X_train):.1%}).")
if svm_clf.kernel == "linear":
    print("  -> SVM tuyến tính cũng có hệ số giải thích được như Logistic.")
else:
    print("  -> SVM kernel phi tuyến không có hệ số trực tiếp (kém giải thích hơn Logistic).")

# ----------------------------------------------------------------------------
# 8. CÁC MẪU BỊ DỰ ĐOÁN SAI
# ----------------------------------------------------------------------------
banner("8. CÁC MẪU BỊ DỰ ĐOÁN SAI")
wrong_lr = set(np.where(preds["Logistic Regression"] != y_test)[0])
wrong_svm = set(np.where(preds["SVM"] != y_test)[0])
print(f"Logistic sai: {len(wrong_lr)} mẫu | SVM sai: {len(wrong_svm)} mẫu")
print(f"  Cả hai cùng sai : {len(wrong_lr & wrong_svm)}")
print(f"  Chỉ Logistic sai: {len(wrong_lr - wrong_svm)}")
print(f"  Chỉ SVM sai     : {len(wrong_svm - wrong_lr)}")

# ----------------------------------------------------------------------------
# 9. DASHBOARD TỔNG HỢP
# ----------------------------------------------------------------------------
banner("9. DASHBOARD TỔNG HỢP")
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
for name in best_models:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    ax.plot(fpr, tpr, lw=2, color=COLORS[name],
            label=f"{name} ({res_df.loc[name, 'ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set(title="ROC", xlabel="FPR", ylabel="TPR")
ax.legend(loc="lower right", fontsize=8)

ax = fig.add_subplot(gs[0, 1])
xb = np.arange(len(metrics_plot))
for i, name in enumerate(best_models):
    ax.bar(xb + (i - 0.5) * 0.38, [res_df.loc[name, m] for m in metrics_plot],
           0.38, color=COLORS[name], label=name)
ax.set_xticks(xb)
ax.set_xticklabels([m.split(" ")[0] for m in metrics_plot], rotation=30, fontsize=8)
ax.set_ylim(0.85, 1.02)
ax.set_title("Chỉ số trên tập test")
ax.legend(fontsize=8, loc="lower right")

ax = fig.add_subplot(gs[0, 2])
ax.boxplot([cv_scores[n]["test_accuracy"] for n in best_models],
           labels=["Logistic", "SVM"], patch_artist=True,
           boxprops=dict(facecolor="#ddd"))
ax.set_title("Accuracy – CV 5×10")

for j, name in enumerate(best_models):
    ax = fig.add_subplot(gs[1, j])
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds[name], display_labels=["Ác tính", "Lành tính"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion – {name}", color=COLORS[name])
    ax.grid(False)

ax = fig.add_subplot(gs[1, 2])
tt = [res_df.loc[n, "Thời gian train (ms)"] for n in best_models]
pt = [res_df.loc[n, "Thời gian predict (ms)"] for n in best_models]
xb = np.arange(2)
ax.bar(xb - 0.2, tt, 0.4, label="Train (ms)", color="#8a8a8a")
ax.bar(xb + 0.2, pt, 0.4, label="Predict (ms)", color="#c9a227")
ax.set_xticks(xb)
ax.set_xticklabels(["Logistic", "SVM"])
ax.set_yscale("log")
ax.set_title("Thời gian (thang log)")
ax.legend(fontsize=8)

fig.suptitle("DASHBOARD: Logistic Regression vs SVM – Breast Cancer Wisconsin",
             fontsize=15, fontweight="bold")
save(fig, "00_dashboard.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 10. KẾT LUẬN TỰ ĐỘNG
# ----------------------------------------------------------------------------
banner("10. KẾT LUẬN")
a, b = "Logistic Regression", "SVM"
for m in ["Accuracy", "Recall (Sensitivity)", "Specificity", "F1-score", "ROC-AUC", "MCC"]:
    va, vb = res_df.loc[a, m], res_df.loc[b, m]
    winner = a if va > vb else b if vb > va else "Hòa"
    print(f"  {m:<22}: Logistic = {va:.4f} | SVM = {vb:.4f}  -> {winner}")
faster = a if res_df.loc[a, "Thời gian train (ms)"] < res_df.loc[b, "Thời gian train (ms)"] else b
print(f"  {'Huấn luyện nhanh hơn':<22}: {faster}")
print("\nGợi ý đọc kết quả:")
print("  • Với dữ liệu nhỏ (569 mẫu) và gần như tách tuyến tính, hai mô hình thường rất sát nhau;")
print("    hãy dựa vào kiểm định t (mục 4) trước khi kết luận mô hình nào 'thật sự' tốt hơn.")
print("  • Trong chẩn đoán ung thư, Recall của lớp ác tính (ít bỏ sót) quan trọng hơn Accuracy.")
print("  • Logistic dễ giải thích (hệ số) và cho xác suất hiệu chuẩn tốt; SVM (RBF) linh hoạt hơn")
print("    với ranh giới phi tuyến nhưng tốn tài nguyên hơn khi dữ liệu lớn.")
print(f"\nTất cả biểu đồ & bảng đã lưu trong thư mục: ./{OUT_DIR}/")