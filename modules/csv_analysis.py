from shiny import module, ui, render, reactive, req
from pathlib import Path
import pandas as pd
from processing import csv_analysis

@module.ui
def csv_analysis_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Input options",
                    ui.input_radio_buttons(
                        "input_type",
                        None,
                        choices={
                            "csv_folder": "Folder of CSVs",
                            "single_csv": "Single CSV",
                        },
                        selected="csv_folder",
                    ),
                    ui.panel_conditional(
                        "input.input_type === 'single_csv'",
                        ui.input_text("csv_path", "CSV path", placeholder="/data/data.csv"),
                        ui.tags.small("Must contain a 'label' column.", style="color:var(--bs-secondary)")
                    ),
                    ui.panel_conditional(
                        "input.input_type === 'csv_folder'",
                        ui.input_text("csv_folder_path", "CSV folder path", placeholder="/data/csvs"),
                        ui.tags.small("All csvs must contain a 'label' column.", style="color:var(--bs-secondary)"),
                        ui.output_data_frame("csvs_table")
                    )
                ),
                open=True,
                id="acc_csv_input",
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Output options",
                    ui.input_text("output_folder", "Output folder", placeholder="/data/output/"),
                    ui.input_checkbox("save_rf_performance", "Save RF performance", value=True),
                    ui.input_checkbox("save_confusion_matrix", "Save prediction matrices", value=True),
                    ui.input_checkbox("save_shap", "Save SHAP graphs", value=True),
                    ui.input_checkbox("sort_shap", "Sort SHAP graphs by feature importance", value=True),
                    ui.input_checkbox("save_pairplots", "Save top feature pairplots",value=True)
                ),
                open=True,
                id="acc_csv_output",
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Parameters",
                    ui.tags.h6("Random Forest", style="margin-top:0.75rem; text-decoration:underline;"),
                    ui.input_numeric("max_depth", "Max tree depth", min=1, max=50, value=5, step=1),
                    ui.input_numeric("n_estimators", "Number of trees", min=10, max=500, value=128, step=1),
                    ui.input_numeric("min_samples_split", "Min samples per split", min=2, max=50, value=20, step=1),
                    ui.input_numeric("min_samples_leaf", "Min samples per leaf", min=2, max=20, value=5, step=1),

                    ui.tags.h6("Cross-validation", style="margin-top:0.75rem; text-decoration:underline;"),
                    ui.input_numeric("n_splits", "Number of splits", min=2, max=100, value=5, step=1),
                    ui.input_numeric("n_repeats", "Number of repeats", min=1, max=20, value=10, step=1)
                ),
                open=True,
                id="acc_rf",
            ),

            ui.input_action_button("run_btn", "Run analysis", class_="btn-primary w-100 mt-2"),
            ui.output_text("run_status"),
            width=290,
            style="overflow-y: auto; max-height: 100vh;"
        ),

        ui.h5("Results"),
        ui.card(ui.card_header("Prediction matrix"), ui.output_plot("plot_conf_matrix"), height="500px"),
        ui.card(ui.card_header("SHAP summary"), ui.output_plot("plot_shap_summary"), height="800px"),
        ui.card(ui.card_header("Pairplot (top 4 SHAP features)"), ui.output_plot("plot_pairplot"), height="900px"),
        fillable=False
    )


@module.server
def csv_analysis_server(input, output, session):

    @reactive.calc
    def params():
        return {
            # Input options
            "input_type": input.input_type(),
            "csv_path": input.csv_path() if input.input_type() == "single_csv" else None,
            "csv_folder_path": input.csv_folder_path() if input.input_type() == "csv_folder" else None,
            # Output options
            "output_folder": input.output_folder(),
            "save_rf_performance": input.save_rf_performance(),
            "save_confusion_matrix": input.save_confusion_matrix(),
            "save_shap": input.save_shap(),
            "sort_shap": input.sort_shap(),
            "save_pairplots": input.save_pairplots(),
            # Parameters
            "max_depth": input.max_depth(),
            "n_estimators": input.n_estimators(),
            "min_samples_split": input.min_samples_split(),
            "min_samples_leaf": input.min_samples_leaf(),
            "n_splits": input.n_splits(),
            "n_repeats": input.n_repeats(),
        }

    results = reactive.value(None)
    csvs = reactive.value(pd.DataFrame())

    @reactive.effect
    def _update_csv_data():
        csv_folder_path = input.csv_folder_path()
        if not csv_folder_path or not Path(csv_folder_path).exists():
            csvs.set(pd.DataFrame())
            return

        csv_files = sorted([f.name for f in Path(csv_folder_path).iterdir() if f.suffix.lower() == ".csv"])
        csvs.set(pd.DataFrame({"File name": csv_files}))

    @render.data_frame
    def csvs_table():
        return render.DataTable(csvs(), width="100%", height="100%", editable=False)

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        p = params()
        # Validate inputs
        if p["input_type"] == "csv_folder":
            if not p["csv_folder_path"]:
                ui.notification_show("Please enter a folder path.", type="error", duration=4)
                return
            if not Path(p["csv_folder_path"]).exists():
                ui.notification_show(f"Folder not found: {p['csv_folder_path']}", type="error", duration=4)
                return
        else:
            if not p["csv_path"]:
                ui.notification_show("Please enter a file path.", type="error", duration=4)
                return
            if not Path(p["csv_path"]).exists():
                ui.notification_show(f"File not found: {p['csv_path']}", type="error", duration=4)
                return

        if any([p["save_rf_performance"], p["save_confusion_matrix"], p["save_shap"], p["save_pairplots"], p["save_distributions"]]) and not p["output_folder"]:
            ui.notification_show("Please enter an output folder path.", type="error", duration=4)
            return

        try:
            ui.notification_show("Processing CSVs", id="processing_notif", duration=None)
            output = csv_analysis.analyze_csv(p)
            results.set(output)
            ui.notification_remove("processing_notif")
            ui.notification_show("Completed", type="default", duration=3)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error", duration=6)

    @render.text
    def run_status():
        r = results.get()
        if r is None:
            return "Waiting to run…"
        return "Processed CSVs"

    @render.plot
    def plot_conf_matrix():
        r = results.get()
        req(r)
        return r["figures"]["confusion_matrix"]

    @render.plot
    def plot_shap_summary():
        r = results.get()
        req(r)
        return r["figures"]["shap_overall"]

    @render.plot
    def plot_pairplot():
        r = results.get()
        req(r)
        return r["figures"]["pairplot"]