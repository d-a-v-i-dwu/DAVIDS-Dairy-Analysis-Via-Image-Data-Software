from skimage.morphology import medial_axis, closing, footprint_rectangle, remove_small_holes, remove_small_objects, disk, opening
from skan import Skeleton, summarize
from skan.csr import PathGraph
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import color
from skimage.measure import label
import seaborn as sns
import os
import cv2
import pandas as pd
from pathlib import Path
import random
import math

random.seed(1)
np.random.seed(1)

def get_images(p: dict, sample_img=False):
    if p["input_type"] == "parent_folder":
        subfolder_paths = [f.path for f in os.scandir(p["parent_folder_path"]) if f.is_dir()]
        if sample_img:
            max_n = 1
        else:
            max_n = p["max_images_subfolder"] or None

        # For each subfolder, only extract up to max_n images
        image_label_pairs = []
        for idx, subfolder_path in enumerate(subfolder_paths):
            images = sorted([(os.path.join(subfolder_path, f), p["subfolder_labels"][idx]) for f in os.listdir(subfolder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])[:max_n]
            image_label_pairs.extend(images)

        return image_label_pairs

    elif p["input_type"] == "folder":
        if sample_img:
            max_n = 1
        else:
            max_n = p["max_images"] or None

        # Only extract up to max_n images
        return sorted([(os.path.join(p["folder_path"], f), p["folder_label"]) for f in os.listdir(p["folder_path"]) if f.lower().endswith(('png', 'jpg', 'jpeg'))])[:max_n]
    
    else:
        return [(p["file_path"], p["file_label"])]


def resize_image(img, resize_to):
    height, width = img.shape[:2]
    side = min(height, width)

    # Center crop the image so it's a square
    start_x = (width - side) // 2
    start_y = (height - side) // 2

    cropped = img[start_y:start_y+side, start_x:start_x+side]
    cropped = cropped.astype(np.uint8)

    img = cv2.resize(cropped, (resize_to, resize_to))
    return img


def create_mask(img, hue_lower, hue_upper, saturation, value):
    # Convert the image to HSV, then normalize
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hue = hsv[:, :, 0] / 179.0
    img_saturation = hsv[:, :, 1] / 255.0
    img_value = hsv[:, :, 2] / 255.0

    # Threshold range
    mask = (img_hue > hue_lower) & (img_hue < hue_upper) & (img_saturation > saturation) & (img_value > value)
    return mask


def clean_mask(mask, open, close, rsh, rso):
    if open != 0:
        mask = opening(mask, disk(open))
    if close != 0:
        mask = closing(mask, footprint_rectangle((close, close)))
    if rsh != 0:
        mask = remove_small_holes(mask, area_threshold=rsh)
    if rso != 0:
        mask = remove_small_objects(mask, min_size=rso)
    
    return mask


def save_mask(image_path, mask, suffix, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    image_name = Path(image_path).stem
    new_img_name = f"{image_name}_{suffix}.png"
    output_path = os.path.join(output_folder_path, new_img_name)

    cv2.imwrite(output_path, mask.astype(np.uint8) * 255)


def save_csvs(df, output_folder, csv_name, multiple_samples):
    if multiple_samples:
        # Store all CSVs in a folder
        output_folder = os.path.join(output_folder, "CSVs")
        os.makedirs(output_folder, exist_ok=True)

        # Save a CSV for each label
        labels = df["label"].unique()
        for sample_label in labels:
            subset = df[df["label"] == sample_label]

            csv_output_path = os.path.join(output_folder, f"{sample_label} Summary.csv")
            subset.to_csv(csv_output_path, index=False)
    
    # Save combined CSV
    csv_output_path = os.path.join(output_folder, f"{csv_name} Summary.csv")
    df.to_csv(csv_output_path, index=False)


def save_graphs(df, output_folder, multiple_samples):
    HIST_KWARGS = {"bins": 15, "alpha": 1, "edgecolor": "black"}
    COLS = 6
    n_attrs = len(FEATURE_COLUMNS)
    n_rows = math.ceil(n_attrs / COLS)

    if multiple_samples:
        output_folder = os.path.join(output_folder, "Graphs")
        os.makedirs(output_folder, exist_ok=True)

        # Feature distribution graphs with boxplots
        fig_dists, axes = plt.subplots(nrows=n_rows, ncols=COLS, figsize=(COLS * 4, n_rows * 3.5))
        axes = axes.flatten()

        for i, col in enumerate(FEATURE_COLUMNS):
            boxplot = sns.boxplot(x="label", y=col, hue="label", data=df, ax=axes[i], fill=False, linewidth=1.2, palette="Set1", fliersize=0)
            stripplot = sns.stripplot(x="label", y=col, hue="label", data=df, ax=axes[i], jitter=True, palette="Set1", alpha=0.5, size=3)
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel('')
            axes[i].set_ylabel(col)

        for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

        fig_dists.suptitle(f"Feature Distributions", fontsize=13, y=1.01)
        fig_dists.tight_layout()
        fig_dists.savefig(os.path.join(output_folder, "feature distributions.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_dists)

        # Histograms of each feature, separated by label
        labels = df["label"].unique()
        n_labels = len(labels)
        palette = dict(zip(labels, plt.cm.tab10.colors[:n_labels]))

        # Separate rows per label
        fig_rows, axs = plt.subplots(n_labels, n_attrs, figsize=(n_attrs * 4, n_labels * 3))
        axs = axs.reshape(n_labels, n_attrs)
        fig_rows.suptitle("Histograms by Label", fontsize=14, y=1.02)

        for row_idx, sample_label in enumerate(labels):
            subset = df[df["label"] == sample_label]
            for col_idx, att in enumerate(FEATURE_COLUMNS):
                ax = axs[row_idx, col_idx]
                ax.hist(subset[att], color=palette[sample_label], **HIST_KWARGS)
                if row_idx == 0:
                    ax.set_title(att, fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(sample_label, fontsize=8)
                ax.tick_params(labelsize=6)

        plt.tight_layout()
        fig_rows.savefig(os.path.join(output_folder, "Label Comparison Graph Separate.png"), bbox_inches="tight")
        plt.close(fig_rows)

        # Density plots of each feature, with labels overlapping
        fig_density, axs = plt.subplots(n_rows, COLS, figsize=(COLS * 5, n_rows * 4))
        fig_density.suptitle("Comparison Density Distributions", fontsize=14)
        axs_flat = axs.flatten()

        for col_idx, att in enumerate(FEATURE_COLUMNS):
            ax = axs_flat[col_idx]
            for sample_label in labels:
                sns.kdeplot(df[df["label"] == sample_label][att], fill=True, label=sample_label, ax=ax, color=palette[sample_label], alpha=0.3)
            ax.tick_params(labelsize=6)
            if col_idx == 0:
                ax.legend(fontsize=7)

        for ax in axs_flat[n_attrs:]:
            ax.set_visible(False)

        plt.tight_layout()
        fig_density.savefig(os.path.join(output_folder, "Label Comparison Graph Combined.png"), bbox_inches="tight")
        plt.close(fig_density)

        # Histograms of each feature, one image saved per label
        for sample_label in labels:
            subset = df[df["label"] == sample_label]
            fig_individual, axs = plt.subplots(n_rows, COLS, figsize=(COLS * 5, n_rows * 4))
            fig_individual.suptitle(f"Histograms - {sample_label}", fontsize=14)
            axs_flat = axs.flatten()

            for ax, att in zip(axs_flat, FEATURE_COLUMNS):
                ax.hist(subset[att], color=palette[sample_label], **HIST_KWARGS)
                ax.set_title(att, fontsize=8)
                ax.tick_params(labelsize=6)

            for ax in axs_flat[n_attrs:]:
                ax.set_visible(False)

            plt.tight_layout()
            fig_individual.savefig(os.path.join(output_folder, f"{sample_label} - Summary Histograms.png"), bbox_inches="tight")
            plt.close(fig_individual)

    # Histogram of each feature for all labels grouped
    fig_overall, axs = plt.subplots(n_rows, COLS, figsize=(COLS * 5, n_rows * 4))
    fig_overall.suptitle("Histograms")
    axs_flat = axs.flatten()

    for ax, att in zip(axs_flat, FEATURE_COLUMNS):
        ax.hist(df[att], **HIST_KWARGS)
        ax.set_title(att, fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axs_flat[n_attrs:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig_overall.savefig(os.path.join(output_folder, "Summary Histograms.png"), bbox_inches="tight")
    plt.close(fig_overall)


def calculate_link_density(skeleton_summary):
    total_branches = len(skeleton_summary)
    if total_branches == 0:
        return 0.0
    
    dangling_ends = skeleton_summary[(skeleton_summary["branch_type"] == 0) | (skeleton_summary["branch_type"] == 1)]
    dangling_loops = skeleton_summary[(skeleton_summary["branch_type"] == 3)]
    
    excluded = set(dangling_ends.index) | set(dangling_loops.index)
    linking_branches = total_branches - len(excluded)
    
    link_density = linking_branches / total_branches
    return link_density


def calculate_fractal_dimension(mask):    
    min_dim = min(mask.shape)
    max_power = int(np.floor(np.log2(min_dim)))
    box_sizes = [2**i for i in range(1, max_power)]
    
    counts = []
    for box_size in box_sizes:
        # Create boxes, checking for non-zero pixels within
        trimmed_rows = (mask.shape[0] // box_size) * box_size
        trimmed_cols = (mask.shape[1] // box_size) * box_size
        trimmed = mask[:trimmed_rows, :trimmed_cols]
        
        reshaped = trimmed.reshape(
            trimmed_rows // box_size, box_size,
            trimmed_cols // box_size, box_size
        )
        non_empty = reshaped.any(axis=(1, 3))
        counts.append(non_empty.sum())
    
    box_sizes = np.array(box_sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    
    # Remove zeros for log
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    
    coeffs = np.polyfit(np.log(1.0 / box_sizes[valid]), np.log(counts[valid]), 1)
    return float(coeffs[0])


def calculate_lacunarity(mask_float):
    box_sizes = [2, 4, 8, 16, 32, 64]
    
    lacunarity_values = []
    
    for box_size in box_sizes:
        h, w = mask_float.shape
        if box_size > h or box_size > w:
            continue
        
        # Pixel sums for every box position
        integral = np.cumsum(np.cumsum(mask_float, axis=0), axis=1)
        
        row_end = h - box_size
        col_end = w - box_size

        br = integral[box_size:, box_size:]
        tl = integral[:row_end, :col_end]
        tr = integral[:row_end, box_size:]
        bl = integral[box_size:, :col_end]
        
        box_sums = (br - tr - bl + tl).flatten()
        
        mean_s = box_sums.mean()
        if mean_s == 0:
            continue
        
        mean_sq = (box_sums ** 2).mean()
        lac = mean_sq / (mean_s ** 2)
        lacunarity_values.append(lac)
    
    if not lacunarity_values:
        return 0.0
    
    return float(np.mean(lacunarity_values))

def summarize_network(mask, skeleton, distances, summary, name, label):
    # Create a PathGraph object for analysis of nodes
    path_graph = PathGraph.from_image(skeleton)

    # Global stats
    skeleton_ids = summary["skeleton_id"]
    num_components = len(np.unique(skeleton_ids))

    local_widths = 2 * distances[skeleton]
    mean_network_width = local_widths.mean()

    network_area = np.count_nonzero(mask)
    network_percentage = (np.count_nonzero(mask) / mask.size)

    link_density = calculate_link_density(summary)
    lacunarity = calculate_lacunarity(mask.astype(np.float32))
    fractal_dimension = calculate_fractal_dimension(mask)
    
    # Branch stats
    branch_lengths = summary["branch_distance"]

    tip_tip_branches = np.sum(summary["branch_type"] == 0)
    tip_junction_branches = np.sum(summary["branch_type"] == 1)
    junction_junction_branches = np.sum(summary["branch_type"] == 2)
    cycles = np.sum(summary["branch_type"] == 3)

    num_branches = tip_tip_branches + tip_junction_branches + junction_junction_branches + cycles
    
    non_cycle = summary[summary["branch_type"] != 3]
    tortuosity = non_cycle["branch_distance"] / non_cycle["euclidean_distance"]
    
    # Node stats
    degrees = path_graph.degrees
    unique_degrees, degree_counts = np.unique(degrees, return_counts=True)

    end_points = unique_degrees == 1
    num_end_points = degree_counts[end_points].sum()
    end_point_density = num_end_points / network_area

    junctions = unique_degrees > 2
    num_junctions = degree_counts[junctions].sum()
    junction_density = num_junctions / network_area

    mean_junction_degree = (unique_degrees[junctions] * degree_counts[junctions]).sum() / num_junctions
    cyclomatic_number = num_branches - num_junctions + num_components
    
    return {
        "name": name,
        "label": label,
        # Global stats
        "num_components": int(num_components),
        "cyclomatic_number": float(cyclomatic_number),
        "mean_network_width": float(mean_network_width),
        "network_percentage": float(network_percentage),
        "tortuosity": float(tortuosity.mean()),
        "link_density": float(link_density),
        "lacunarity": float(lacunarity),
        "fractal_dimension": float(fractal_dimension),
        # Branch stats
        "num_branches": int(len(branch_lengths)),
        "tip_tip_branches": int(tip_tip_branches),
        "tip_junction_branches": int(tip_junction_branches),
        "junction_junction_branches": int(junction_junction_branches),
        "cycles": int(cycles),
        "avg_branch_length": float(np.mean(branch_lengths)),
        "total_branch_length": float(np.sum(branch_lengths)),
        "max_branch_length": float(np.max(branch_lengths)),
        # Junction stats
        "num_end_points": int(num_end_points),
        "end_point_density": float(end_point_density),
        "num_junctions": int(num_junctions),
        "junction_density": float(junction_density),
        "unique_degrees": unique_degrees.tolist(),
        "degree_counts": degree_counts.tolist(),
        "avg_junction_degree": float(mean_junction_degree),
    }

def process_sample_image(p: dict):
    image_label_pairs = get_images(p, sample_img=True)
    img = cv2.imread(image_label_pairs[0][0])

    if p["gaussian_sigma"] != 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=p["gaussian_sigma"])

    protein_mask = create_mask(img, p["hue_lower"], p["hue_upper"], p["saturation"], p["value"])
    mask = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
    mask[protein_mask, :] = [1.0]
    mask = np.any(mask, axis=-1)

    mask = clean_mask(mask, p["opening"], p["closing"], p["remove_small_holes"], p["remove_small_objects"])

    med_axis = medial_axis(mask, rng=1)
    skeleton = Skeleton(med_axis)
    summary = summarize(skeleton, separator="_")

    if p["prune_branches"] != 0:
        short_edges = summary[(summary["branch_type"] == 1) & (summary["branch_distance"] < p["prune_branches"])].index
        skeleton = skeleton.prune_paths(short_edges)
    
    skeleton_img = skeleton.skeleton_image

    if p["resize_to"] != 0:
        img = resize_image(img, p["resize_to"])
        mask = resize_image(mask, p["resize_to"])
        skeleton_img = resize_image(skeleton_img, p["resize_to"])

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0,0].imshow(rgb_img)
    ax[0,0].set_title("Original Image")
    ax[0,0].axis("off")

    ax[0,1].imshow(rgb_img)
    ax[0,1].imshow(skeleton_img, cmap="inferno", alpha=0.3)
    ax[0,1].set_title("Overlaid Skeleton")
    ax[0,1].axis("off")

    ax[1,0].imshow(rgb_img)
    ax[1,0].imshow(mask, cmap="inferno", alpha=0.3)
    ax[1,0].set_title("Overlaid Mask")
    ax[1,0].axis("off")

    ax[1,1].imshow(mask, cmap="gray")
    ax[1,1].set_title("Base mask")
    ax[1,1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    return fig


def process_images(p: dict):
    try:
        if p["output_folder"] is not None:
            os.makedirs(p["output_folder"], exist_ok=True)
        
        # Get all the input files
        image_label_pairs = get_images(p)
        
        all_summaries = []
        
        for image_label_pair in image_label_pairs:
            image_path = image_label_pair[0]
            sample_label = image_label_pair[1]
            img = cv2.imread(image_path)

            # Apply Gaussian filter to smooth the image before masking
            if p["gaussian_sigma"] != 0:
                img = cv2.GaussianBlur(img, (0, 0), sigmaX=p["gaussian_sigma"])

            # Extract the protein network
            protein_mask = create_mask(img, p["hue_lower"], p["hue_upper"], p["saturation"], p["value"])
            mask = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
            mask[protein_mask, :] = [1.0]
            mask = np.any(mask, axis=-1)

            # Clean the network to remove noise
            mask = clean_mask(mask, p["opening"], p["closing"], p["remove_small_holes"], p["remove_small_objects"])

            if p["resize_to"] != 0:
                mask = resize_image(mask, p["resize_to"])
                mask = mask.astype(bool)

            # Skeletonize / save the image
            if p["save_skeletons"] or p["save_components"] or p["save_csvs"] or p["save_graphs"]:
                med_axis, distances = medial_axis(mask, return_distance=True ,rng=1)

                skeleton = Skeleton(med_axis)
                summary = summarize(skeleton, separator="_")

                # Prune branches below a certain size
                if p["prune_branches"] != 0:
                    short_edges = summary[(summary["branch_type"] == 1) & (summary["branch_distance"] < p["prune_branches"])].index
                    skeleton = skeleton.prune_paths(short_edges)
                    summary = summarize(skeleton, separator="_")
                
                skeleton_img = skeleton.skeleton_image

                # Save the skeleton image
                if p["save_skeletons"]:
                    skeleton_output_folder_path = os.path.join(p["output_folder"], "Skeletons")
                    if p["input_type"] == "parent_folder":
                        skeleton_output_folder_path = os.path.join(skeleton_output_folder_path, sample_label)

                    save_mask(image_path, skeleton_img, "skeleton", skeleton_output_folder_path)

                # Graph the connected components in different colours
                if p["save_components"]:
                    labelled_regions = label(med_axis, connectivity=2)
                    coloured_components = color.label2rgb(labelled_regions, image=med_axis, bg_label=0)

                    components_output_folder_path = os.path.join(p["output_folder"], "Coloured Components")
                    if p["input_type"] == "parent_folder":
                        components_output_folder_path = os.path.join(components_output_folder_path, sample_label)

                    save_mask(image_path, coloured_components, "coloured_components", components_output_folder_path)

                # Generate and save network summary CSVs to a list
                if p["save_csvs"] or p["save_graphs"]:
                    network_summary = summarize_network(mask, skeleton_img, distances, summary, Path(image_path).stem, sample_label)
                    all_summaries.append(network_summary)
                
            # If an output folder was specified, save the protein network image
            if p["save_masks"]:
                mask_output_folder_path = os.path.join(p["output_folder"], "Masks")
                if p["input_type"] == "parent_folder":
                    mask_output_folder_path = os.path.join(mask_output_folder_path, sample_label)

                save_mask(image_path, mask, "mask", mask_output_folder_path)
        
        df = pd.DataFrame(all_summaries)

        if p["save_csvs"]:
            csv_name = (
                Path(p["parent_folder_path"]).name if p["parent_folder_path"] else
                Path(p["folder_path"]).name if p["folder_path"] else
                Path(p["file_path"]).stem
            )
            save_csvs(df, p["output_folder"], csv_name, p["input_type"] == "parent_folder")

        # Plot the histogram distributions
        if p["save_graphs"]:
            save_graphs(df, p["output_folder"], p["input_type"] == "parent_folder")

    # Ensure all graphs are closed no matter what 
    finally:
        plt.close("all")

FEATURE_COLUMNS = ["num_components", "cyclomatic_number", "mean_network_width", "network_percentage", "tortuosity", "link_density", "lacunarity", "fractal_dimension", "num_branches", "tip_tip_branches", "tip_junction_branches", "junction_junction_branches", "cycles", "avg_branch_length", "total_branch_length", "max_branch_length", "num_end_points", "end_point_density", "num_junctions", "junction_density", "avg_junction_degree"]

# Test call
if __name__ == "__main__":
    p = {
        "input_type": "folder",
        "parent_folder_path": "",
        "subfolder_labels": [],
        "folder_path": "",
        "folder_label": "",
        "file_path": None,
        "file_label": None,
        # Output options
        "max_images_subfolder": 0,
        "max_images": 0,
        "resize_to": 0,
        "output_folder": "",
        "save_masks": True,
        "save_skeletons": True,
        "save_components": True,
        "save_csvs": True,
        "save_graphs": True,
        # Parameters
        "gaussian_sigma": 0,
        "hue_lower": 0.2,
        "hue_upper": 0.45,
        "saturation": 0.4,
        "value": 0.27,
        "remove_small_holes": 0,
        "remove_small_objects": 0,
        "closing": 0,
        "opening": 0,
        "prune_branches": 0,
    }

    process_images(p)

