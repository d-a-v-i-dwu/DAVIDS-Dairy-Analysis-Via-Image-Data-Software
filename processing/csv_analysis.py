from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_predict
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import os
from pathlib import Path
import seaborn as sns


def rf_performance(X, y, save_rf_perfomrance, output_folder, save_confusion_matrix, n_splits, n_repeats, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    acc_scores = []
    train_acc_scores = []
    reports = []

    # Train random forest and save stats for each repeat
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        rf.fit(X_train, y_train)

        # Evaluate RF on test set
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

        # Evaluate RF on training set
        y_train_pred = rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_acc_scores.append(train_acc)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports.append(report)

        y_pred_cv = cross_val_predict(rf, X, y, cv=10)
        cm = confusion_matrix(y, y_pred_cv)

    # Create confusion matrix
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot(ax=ax_cm)
    if save_confusion_matrix:
        fig_cm.savefig(os.path.join(output_folder, "RF Confusion matrix.png"), bbox_inches="tight")
    plt.close(fig_cm)

    # Average performance across folds
    lines = []
    lines.append(f"RF Parameters = {{n_estimators: {n_estimators}, max_depth: {max_depth}, min_samples_split: {min_samples_split}}}")
    lines.append(f"{n_repeats} Repeats, {n_splits} Splits")
    lines.append("")
    lines.append(f"Testing accuracy  = {{mean: {np.mean(acc_scores):.5f}, median: {np.median(acc_scores):.5f}, std: {np.std(acc_scores):.5f}}}")
    lines.append(f"Training accuracy = {{mean: {np.mean(train_acc_scores):.5f}, median: {np.mean(train_acc_scores):.5f}, std: {np.std(train_acc_scores):.5f}}}")
    lines.append("")

    for sample in rf.classes_:
        f1_scores = [x[sample]['f1-score'] for x in reports]
        lines.append(f"F1 score {sample:<9} = {{mean: {np.mean(f1_scores):.5f}, median: {np.median(f1_scores):.5f}, std: {np.std(f1_scores):.5f}}}")

    performance_text = "\n".join(lines)

    if save_rf_perfomrance:
        lines.append("")
        lines.append(f"test_acc_scores  = {acc_scores}")
        lines.append(f"train_acc_scores = {train_acc_scores}")
        lines.append("")

        for sample in rf.classes_:
            f1_scores = [x[sample]['f1-score'] for x in reports]
            lines.append(f"{sample.strip()} f1_score = {f1_scores}")

        with open(os.path.join(output_folder, "RF Performance.txt"), "w") as f:
            f.write("\n".join(lines))

    return {
        "performance_text": performance_text,
        "confusion_matrix": fig_cm,
    }


def shapley_values(X, y, output_folder, sort, save_graphs, input_name, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if output_folder:
        shapley_folder = os.path.join(output_folder, "Shapley Graphs")
        os.makedirs(shapley_folder, exist_ok=True)
    else:
        shapley_folder = None

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    shap_values_arr = explainer.shap_values(X_train)

    # Overall bar plot
    shap.summary_plot(shap_values_arr, feature_names=FEATURE_COLUMNS, class_names=rf.classes_, plot_type="bar", show_values_in_legend=True, rng=42, show=False, sort=sort)
    plt.suptitle(f"{input_name} SHAP Summary", fontsize=14)
    fig_bar = plt.gcf()
    if save_graphs and shapley_folder:
        fig_bar.savefig(os.path.join(shapley_folder, "Shapley Feature Importance Bar Plot.png"))
    plt.close(fig_bar)

    # Get top 4 features by mean absolute SHAP for dependence plots
    mean_abs_shap = np.abs(shap_values_arr).mean(axis=(0, 2))
    top_indices = np.argsort(mean_abs_shap)[::-1][:4]
    top_features = [FEATURE_COLUMNS[i] for i in top_indices]

    # Per-class plots
    if save_graphs:
        for i, label in enumerate(rf.classes_):
            class_shap = shap_values_arr[:, :, i]
            class_folder = os.path.join(shapley_folder, label) if shapley_folder else None
            if class_folder:
                os.makedirs(class_folder, exist_ok=True)

            # Beeswarm summary plot
            plot_name = f"{label} SHAP Summary"
            shap.summary_plot(class_shap, X_train, feature_names=FEATURE_COLUMNS, show=False, sort=sort, rng=42)
            plt.suptitle(plot_name, y=1.02)
            fig_summary = plt.gcf()
            fig_summary.savefig(os.path.join(class_folder, "Summary Plot.png"), bbox_inches="tight")
            plt.close(fig_summary)

            # Top 4 feature dependence plots
            dep_folder = os.path.join(class_folder, "Dependence Plots") if class_folder else None
            if dep_folder:
                os.makedirs(dep_folder, exist_ok=True)

            for feature in top_features:
                shap.dependence_plot(feature, class_shap, X_train, feature_names=FEATURE_COLUMNS, show=False)
                plt.suptitle(f"{label} SHAP Dependence: {feature}", y=1.02)
                fig_dep = plt.gcf()
                fig_dep.savefig(os.path.join(dep_folder, f"{feature}.png"), bbox_inches="tight")
                plt.close(fig_dep)

    return {
        "shap_overall": fig_bar,
        "top_features": top_features
    }


def create_pairplot(X, y, top_features, input_name, output_folder, save_graph):
    plot_df = X[top_features].copy()
    plot_df["label"] = y.values

    pairplot = sns.pairplot(plot_df, hue="label")
    pairplot.figure.suptitle(f"{input_name} Top 4 SHAP Features Pairplot", y=1.02)
    
    pairplot._legend.remove()
    pairplot.figure.legend(handles=pairplot._legend_data.values(), labels=pairplot._legend_data.keys(), bbox_to_anchor=(1.02, 1.0))
    pairplot.figure.tight_layout()

    if save_graph:
        pairplot.figure.savefig(os.path.join(output_folder, f"Top 4 Features Pairplot.png"), bbox_inches="tight")

    return pairplot.figure


def analyze_csv(p: dict):
    try:
        if p["input_type"] == "csv_folder":
            df = pd.concat([pd.read_csv(os.path.join(p["csv_folder_path"], f)) for f in os.listdir(p["csv_folder_path"]) if f.lower().endswith(".csv")], ignore_index=True)
            input_name = Path(p["csv_folder_path"]).name
        else:
            df = pd.read_csv(p["csv_path"])
            input_name = Path(p["csv_path"]).stem

        if p["output_folder"]:
            os.makedirs(p["output_folder"], exist_ok=True)

        df.drop(["name", "branch_types", "branch_type_counts", "unique_degrees", "degree_counts", "min_branch_length"], inplace=True, axis=1, errors="ignore")
        X = df.drop(columns=["label"])
        y = df["label"].astype(str)

        rf_outputs = rf_performance(X, y, p["save_rf_performance"], p["output_folder"], p["save_confusion_matrix"], p["n_splits"], p["n_repeats"], p["n_estimators"], p["max_depth"], p["min_samples_split"], p["min_samples_leaf"])
        shap_outputs = shapley_values(X, y, p["output_folder"], p["sort_shap"], p["save_shap"], input_name, p["n_estimators"], p["max_depth"], p["min_samples_split"], p["min_samples_leaf"])
        pairplot = create_pairplot(X, y, shap_outputs["top_features"], input_name, p["output_folder"], p["save_pairplots"])

        return {
            "text": {
                "performance": rf_outputs["performance_text"],
            },
            "figures": {
                "confusion_matrix": rf_outputs["confusion_matrix"],
                "shap_overall": shap_outputs["shap_overall"],
                "pairplot": pairplot
            },
        }
    finally:
        plt.close("all")

FEATURE_COLUMNS = ["num_components", "cyclomatic_number", "mean_network_width", "network_percentage", "tortuosity", "link_density", "lacunarity", "fractal_dimension", "num_branches", "tip_tip_branches", "tip_junction_branches", "junction_junction_branches", "cycles", "avg_branch_length", "total_branch_length", "max_branch_length", "num_end_points", "end_point_density", "num_junctions", "junction_density", "avg_junction_degree"]

# Test call
if __name__ == "__main__":
    p = {
        "input_type": "single_csv",
        "csv_path": "",
        "csv_folder_path": "",
        "output_folder": "",
        "save_rf_performance": True,
        "save_confusion_matrix": True,
        "save_shap": True,
        "sort_shap": True,
        "save_pairplots": True,
        "max_depth": 5,
        "n_estimators": 128,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "n_splits": 5,
        "n_repeats": 10,
    }

    analyze_csv(p)