from shiny import module, ui, render, reactive, req
from pathlib import Path
from processing import mask_generation
import os
import pandas as pd

@module.ui
def mask_generation_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Input options",
                    ui.input_radio_buttons(
                        id="input_type",
                        label=None,
                        choices={"parent_folder": "Parent folder", "folder": "Single folder", "file": "Single file"},
                        selected="parent_folder",
                    ),
                    ui.panel_conditional(
                        "input.input_type === 'parent_folder'",
                        ui.input_text("parent_folder_path", "Folder path", placeholder="/data/subfolders/"),
                        ui.output_data_frame("subfolders"),
                        ui.tags.small("You may edit the labels column to change the label for outputs"),
                    ),
                    ui.panel_conditional(
                        "input.input_type === 'folder'",
                        ui.input_text("folder_path", "Folder path", placeholder="/data/images/"),
                        ui.input_text("folder_label", "Sample label", placeholder="Sample A"),
                    ),
                    ui.panel_conditional(
                        "input.input_type === 'file'",
                        ui.input_text("file_path", "File path", placeholder="/data/image"),
                        ui.input_text("image_label", "Image label", placeholder="Image A"),
                    ),
                ),
                open=True,
                id="acc_mask_input"
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Output options",
                    ui.panel_conditional(
                        "input.input_type === 'parent_folder'",
                        ui.div(
                            ui.input_numeric("max_images_subfolders", "Max images per subfolder", min=1, max=500, value=0, step=1),
                            ui.tags.small("Leave max images as 0 to process all images"),
                            style="margin-bottom:0.75rem;"
                        ),
                    ),
                    ui.panel_conditional(
                        "input.input_type === 'folder'",
                        ui.div(
                            ui.input_numeric("max_images", "Max images", min=1, max=500, value=0, step=1),
                            ui.tags.small("Leave max images as 0 to process all images"),
                            style="margin-bottom:0.75rem;"
                        ),
                    ),
                    ui.div(
                        ui.input_numeric("resize_to", "Resize to (px)", min=128, max=2048, value=0, step=1),
                        ui.tags.small("Leave resize size as 0 to process images at their input size"),
                        style="margin-bottom:0.75rem;"
                    ),

                    ui.input_text("output_folder", "Output folder", placeholder="/data/output/"),
                    ui.input_checkbox("save_masks", "Save masks", value=True),
                    ui.input_checkbox("save_skeletons", "Save skeletons", value=True),
                    ui.input_checkbox("save_components", "Save components", value=True),
                    ui.input_checkbox("save_csvs", "Save CSVs", value=True),
                    ui.panel_conditional(
                        "input.input_type !== 'file'",
                        ui.input_checkbox("save_graphs", "Save CSV graphs", value=True),
                    )
                ),
                open=True,
                id="acc_output_opts"
            ),

            ui.accordion(
                ui.accordion_panel(
                    "Parameters",
                    ui.input_numeric("gaussian_sigma", "Gaussian Blur Sigma", min=0, max=10, value=0, step=0.01),
                    ui.tags.h6("HSV thresholds", style="text-decoration: underline;"),
                    ui.input_numeric("hue_lower", "Hue Lower Bound", min=0, max=1, value=0.2, step=0.01 ),
                    ui.input_numeric("hue_upper", "Hue Upper Bound", min=0, max=1, value=0.45, step=0.01),
                    ui.input_numeric("saturation", "Saturation", min=0, max=1, value=0.4, step=0.01),
                    ui.input_numeric("value", "Value", min=0, max=1, value=0.27, step=0.01),
                    ui.tags.h6("Morphological Operations", style="margin-top:0.75rem; text-decoration: underline"),
                    ui.input_numeric("opening", "Opening", min=1, max=20, value=5, step=1),
                    ui.input_numeric("closing", "Closing", min=1, max=20, value=0, step=1),
                    ui.input_numeric("remove_small_holes", "Remove small holes", min=0, max=100, value=0, step=1),
                    ui.input_numeric("remove_small_objects", "Remove small objects", min=0, max=100, value=0, step=1),
                    ui.input_numeric("prune_branches", "Prune branches", min=0, max=100, value=4, step=1),
                ),
                open=True,
                id="acc_parameters"
            ),

            ui.input_action_button("run_btn", "Run analysis", class_="btn-primary w-100 mt-2"),
            ui.output_text("run_status"),
            width=290,
            style="overflow-y: auto; max-height: 100vh;"

        ),

        ui.card(
            ui.card_header("Preview (first image)"),
            ui.output_plot("preview_sample"),
            height="1000px"
        ),
        fillable=False
    )


@module.server
def mask_generation_server(input, output, session):

    @reactive.calc
    def params():
        return {
            # Input options
            "input_type": input.input_type(),
            "parent_folder_path": input.parent_folder_path() if input.input_type() == "parent_folder" else None,
            "subfolder_labels": subfolder_labels() if input.input_type() == "parent_folder" else [],
            "folder_path": input.folder_path() if input.input_type() == "folder" else None,
            "folder_label": input.folder_label() if input.input_type() == "folder" else None,
            "file_path": input.file_path() if input.input_type() == "file" else None,
            "image_label": input.image_label() if input.input_type() == "file" else None,
            # Output options
            "max_images_subfolder": input.max_images_subfolders() if input.input_type() == "parent_folder" else 0,
            "max_images": input.max_images() if input.input_type() == "folder" else 0,
            "resize_to": input.resize_to(),
            "output_folder": input.output_folder(),
            "save_masks": input.save_masks(),
            "save_skeletons": input.save_skeletons(),
            "save_components": input.save_components(),
            "save_csvs": input.save_csvs(),
            "save_graphs": input.save_graphs() if input.input_type() != "file" else False,
            # Parameters
            "gaussian_sigma": input.gaussian_sigma(),
            "hue_upper": input.hue_upper(),
            "hue_lower": input.hue_lower(),
            "saturation": input.saturation(),
            "value": input.value(),
            "remove_small_holes": input.remove_small_holes(),
            "remove_small_objects": input.remove_small_objects(),
            "closing": input.closing(),
            "opening": input.opening(),
            "prune_branches": input.prune_branches(),
        }

    results = reactive.value(None)

    subfolder_data = reactive.value(pd.DataFrame())

    @reactive.effect
    def _update_subfolder_data():
        parent_folder_path = input.parent_folder_path()
        if not parent_folder_path or not Path(parent_folder_path).exists():
            subfolder_data.set(pd.DataFrame())
            return
        
        subfolder_paths = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]
        data = [{
            "Subfolder": os.path.basename(os.path.normpath(path)),
            "Images": len([f for f in os.listdir(path) if f.lower().endswith(("png","jpg","jpeg"))]),
            "Label": os.path.basename(os.path.normpath(path))
        } for path in subfolder_paths]
        subfolder_data.set(pd.DataFrame(data))

    @render.data_frame
    def subfolders():
        return render.DataTable(subfolder_data(), width="100%", height="100%", editable=True)

    @reactive.calc
    def subfolder_labels():
        edited = subfolders.data_view()
        return edited["Label"].tolist() if not edited.empty else []

    @reactive.effect
    @reactive.event(input.run_btn)
    def _run():
        p = params()
        if p["input_type"] == "parent_folder":
            if not p["parent_folder_path"]:
                ui.notification_show("Please enter a folder path.", type="error", duration=4)
                return
            if not Path(p["parent_folder_path"]).exists():
                ui.notification_show(f"Folder not found: {p['parent_folder_path']}", type="error", duration=4)
                return
        elif p["input_type"] == "folder":
            if not p["folder_path"]:
                ui.notification_show("Please enter a folder path.", type="error", duration=4)
                return
            if not Path(p["folder_path"]).exists():
                ui.notification_show(f"Folder not found: {p['folder_path']}", type="error", duration=4)
                return
        else:
            if not p["file_path"]:
                ui.notification_show("Please enter a file path.", type="error", duration=4)
                return
            if not Path(p["file_path"]).exists():
                ui.notification_show(f"File not found: {p['file_path']}", type="error", duration=4)
                return

        if not p["output_folder"]:
            ui.notification_show("Please enter an output folder path.", type="error", duration=4)
            return

        ui.notification_show("Processing images", id="processing_imgs", duration=None)
        mask_generation.process_images(p)
        ui.notification_remove("processing_imgs")
        
        ui.notification_show("Completed", type="message", duration=3)

    # Status text
    @render.text
    def run_status():
        if results.get() is None:
            return "Waiting to run…"
        r = results.get()
        return f"Processed {len(r['images'])} image(s)."

    # Image outputs
    @reactive.calc
    def sample_images():
        p = params()
        # Require a valid path before attempting to process
        if p["input_type"] == "parent_folder":
            req(p["parent_folder_path"])
        elif p["input_type"] == "folder":
            req(p["folder_path"])
        else:
            req(p["file_path"])
        return mask_generation.process_sample_image(p)

    @render.plot
    def preview_sample():
        return sample_images()