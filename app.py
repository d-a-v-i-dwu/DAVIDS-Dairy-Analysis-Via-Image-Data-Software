from shiny import App, ui
from modules.mask_generation import mask_generation_ui, mask_generation_server
from modules.csv_analysis import csv_analysis_ui, csv_analysis_server

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Mask generation",
        mask_generation_ui("mask"),
    ),
    ui.nav_panel(
        "CSV analysis",
        csv_analysis_ui("csv"),
    ),
    title="Protein Analysis Tool",
    id="main_tabs",
)


def server(input, output, session):
    mask_generation_server("mask")
    csv_analysis_server("csv")

app = App(app_ui, server)