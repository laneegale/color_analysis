import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageCms
import tempfile
import os
import io
import importlib.util

from src.analyzer import AppliedImageAnalyzer, IT8Analyzer
from streamlit_image_coordinates import streamlit_image_coordinates

IMAGE_TYPES = ["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"]

# Default values for different modes
MODE_DEFAULTS = {
    "Applied Image": {
        "box_size": 12,
        "stride_x": 55,
        "stride_y": 44,
        "rows": 4,
        "cols": 6,
    },
    "IT8": {
        "box_size": 20,
        "stride_x": 40,
        "stride_y": 40,
        "rows": 12,  # Fixed: A-L
        "cols": 22,  # Fixed: 1-22
    },
    "Apply": {
        "box_size": 12,
        "stride_x": 55,
        "stride_y": 44,
        "rows": 4,
        "cols": 6,
    },
}

def resize_for_display(img: Image.Image, max_width=600):
    w, h = img.size
    if w <= max_width:
        return img, 1.0
    scale = max_width / w
    return img.resize((max_width, int(h * scale)), Image.BICUBIC), scale

def apply_icc(img_np, icc_bytes):
    img = Image.fromarray(img_np)
    with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
        f.write(icc_bytes)
        icc_path = f.name
    srgb = ImageCms.createProfile("sRGB")
    src_profile = ImageCms.ImageCmsProfile(icc_path)
    img_icc = ImageCms.profileToProfile(img, src_profile, srgb, outputMode=img.mode)
    return np.array(img_icc)

def load_py_calibrator(py_bytes):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(py_bytes)
        py_path = f.name
    spec = importlib.util.spec_from_file_location("user_calib", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "calibrate"):
        raise ValueError("Python file must define a calibrate(img_np) function")
    return module.calibrate

def generate_boxes(analyzer, settings, mode):
    """Generate bounding boxes with current settings"""
    if settings["bbox_origin_grid"] is None:
        return None
    
    origin = (
        settings["bbox_origin_grid"][0] + settings["offset_x"],
        settings["bbox_origin_grid"][1] + settings["offset_y"]
    )
    
    ca_origin = None
    if settings["bbox_origin_ca"]:
        ca_origin = (
            settings["bbox_origin_ca"][0] + settings["offset_x"],
            settings["bbox_origin_ca"][1] + settings["offset_y"]
        )
    
    boxes = analyzer._generate_bounding_boxes(
        origin=origin,
        rows=settings["rows"],
        cols=settings["cols"],
        box_w=settings["box_size"],
        box_h=settings["box_size"],
        stride_x=settings["stride_x"],
        stride_y=settings["stride_y"],
        anchor="center",
        ca_origin=ca_origin
    )
    return boxes

def get_box_settings_hash():
    """Get hash of current box settings to detect changes"""
    return (
        st.session_state.get("box_size"),
        st.session_state.get("stride_x"),
        st.session_state.get("stride_y"),
        st.session_state.get("rows"),
        st.session_state.get("cols"),
        st.session_state.get("rotate_angle"),
        st.session_state.get("offset_x"),
        st.session_state.get("offset_y"),
        st.session_state.get("bbox_origin_grid"),
        st.session_state.get("bbox_origin_ca"),
    )

st.set_page_config(layout="wide")
st.title("Calibration Slide Analyzer")

# -------- Session State Init --------
defaults = {
    "bbox_origin_grid": None,
    "bbox_origin_ca": None,
    "custom_boxes": None,
    "display_scale": 1.0,
    "offset_x": 0,
    "offset_y": 0,
    "box_size": 12,
    "stride_x": 55,
    "stride_y": 44,
    "rows": 4,
    "cols": 6,
    "rotate_angle": 0.0,
    "last_click_coords": None,
    "last_box_settings_hash": None,
    "last_mode": None,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------- Mode Toggle --------
mode = st.radio("Analysis Mode", options=["Applied Image", "IT8", "Apply"], horizontal=True)

# Update defaults when mode changes
if st.session_state.last_mode != mode:
    mode_defaults = MODE_DEFAULTS[mode]
    st.session_state.box_size = mode_defaults["box_size"]
    st.session_state.stride_x = mode_defaults["stride_x"]
    st.session_state.stride_y = mode_defaults["stride_y"]
    st.session_state.rows = mode_defaults["rows"]
    st.session_state.cols = mode_defaults["cols"]
    st.session_state.last_mode = mode

# -------- File Uploads --------
col1, col2, col3, col4 = st.columns(4)

with col1:
    if mode == "Applied Image":
        uploaded_img = st.file_uploader("Upload Applied Image Slide (.tif)", type=IMAGE_TYPES)
    elif mode == "IT8":
        uploaded_img = st.file_uploader("Upload IT8 Image Slide (.tif)", type=IMAGE_TYPES)
    else:
        uploaded_img = st.file_uploader("Upload H&E Image Slide (.tif)", type=IMAGE_TYPES)

with col2:
    uploaded_ref = st.file_uploader("Upload IT8 Reference File (.txt)", type=["txt"]) if mode == "IT8" else None

with col3:
    uploaded_icc = st.file_uploader("ICC file", type=["icm", "icc"])

with col4:
    uploaded_py = st.file_uploader("Py transform file", type=["py"], help="Must contain 'calibrate(img_np)' function")

# -------- Proceed only if required files exist --------
ready = uploaded_img and (mode in ["Applied Image", "Apply"] or uploaded_ref)

if ready:
    state_key = (uploaded_img.name, uploaded_ref.name if uploaded_ref else None, mode)
    if st.session_state.get("last_state") != state_key:
        st.session_state.transformed_img = None
        st.session_state.df_transformed = None
        st.session_state.bbox_origin_grid = None
        st.session_state.bbox_origin_ca = None
        st.session_state.custom_boxes = None
        st.session_state.offset_x = 0
        st.session_state.offset_y = 0
        st.session_state.last_click_coords = None
        st.session_state.last_box_settings_hash = None
        st.session_state.last_state = state_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_img:
        tmp_img.write(uploaded_img.read())
        image_path = tmp_img.name

    ref_path = None
    if uploaded_ref:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_ref:
            tmp_ref.write(uploaded_ref.read())
            ref_path = tmp_ref.name

    if mode == "IT8":
        analyzer = IT8Analyzer(image_path, ref_path)
    else:
        analyzer = AppliedImageAnalyzer(image_path)

    color_hex_col = lambda val: f"background-color: {val};"

    # ================= BOUNDING BOX GENERATOR =================
    with st.expander("Bounding Box Generator", expanded=True):
        if mode == "IT8":
            # IT8 has fixed grid structure, only show relevant controls
            c1, c2, c3 = st.columns(3)
            st.session_state.box_size = c1.number_input("Box Size", 1, 500, st.session_state.box_size)
            st.session_state.stride_x = c2.number_input("Stride X", 1, 500, st.session_state.stride_x)
            st.session_state.stride_y = c3.number_input("Stride Y", 1, 500, st.session_state.stride_y)
            
            st.caption("ℹ️ IT8 grid is fixed: 12 rows (A-L) × 22 columns + 24 grayscale patches")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            st.session_state.box_size = c1.number_input("Box Size", 1, 500, st.session_state.box_size)
            st.session_state.stride_x = c2.number_input("Stride X", 1, 500, st.session_state.stride_x)
            st.session_state.stride_y = c3.number_input("Stride Y", 1, 500, st.session_state.stride_y)
            st.session_state.rows = c4.number_input("Rows", 1, 50, st.session_state.rows)
            st.session_state.cols = c5.number_input("Cols", 1, 50, st.session_state.cols)
        
        st.session_state.rotate_angle = st.slider("Rotate", -10.0, 10.0, st.session_state.rotate_angle, 0.1)

    # Apply rotation
    analyzer.rotate(st.session_state.rotate_angle)

    # ================= IMAGES ROW =================
    img_col1, ctrl_col, img_col2 = st.columns([3, 1, 3])

    with ctrl_col:
        st.subheader("Controls")
        
        # Anchor point selection
        st.write("**Set Anchor**")
        if mode == "IT8":
            anchor_mode = st.radio("Click to set:", ["Grid (A1)", "Grayscale (GS0)"], horizontal=False, key="anchor_mode", label_visibility="collapsed")
        else:
            anchor_mode = st.radio("Click to set:", ["Grid (A1)", "CA"], horizontal=False, key="anchor_mode", label_visibility="collapsed")
        
        # Display anchor points
        if st.session_state.bbox_origin_grid:
            st.success(f"Grid: {st.session_state.bbox_origin_grid}")
        else:
            st.warning("Grid not set")
        
        if st.session_state.bbox_origin_ca:
            if mode == "IT8":
                st.success(f"GS: {st.session_state.bbox_origin_ca}")
            else:
                st.success(f"CA: {st.session_state.bbox_origin_ca}")
        else:
            if mode == "IT8":
                st.caption("GS not set")
            else:
                st.caption("CA not set")
        
        st.divider()
        
        # Movement controls
        st.write("**Move Boxes**")
        move_step = st.number_input("Step (px)", 1, 100, 5, key="move_step")
        
        # Up button
        up_cols = st.columns([1, 2, 1])
        with up_cols[1]:
            if st.button("⬆️", key="move_up", use_container_width=True):
                st.session_state.offset_y -= move_step
                st.rerun()
        
        # Left and Right buttons
        lr_cols = st.columns(3)
        with lr_cols[0]:
            if st.button("⬅️", key="move_left", use_container_width=True):
                st.session_state.offset_x -= move_step
                st.rerun()
        with lr_cols[2]:
            if st.button("➡️", key="move_right", use_container_width=True):
                st.session_state.offset_x += move_step
                st.rerun()
        
        # Down button
        down_cols = st.columns([1, 2, 1])
        with down_cols[1]:
            if st.button("⬇️", key="move_down", use_container_width=True):
                st.session_state.offset_y += move_step
                st.rerun()
        
        st.divider()
        st.write("**Scale Boxes**")
        
        scale_cols = st.columns(2)
        with scale_cols[0]:
            if st.button("➖ 10%", key="scale_down", use_container_width=True):
                st.session_state.box_size = max(1, int(st.session_state.box_size * 0.9))
                st.session_state.stride_x = max(1, int(st.session_state.stride_x * 0.9))
                st.session_state.stride_y = max(1, int(st.session_state.stride_y * 0.9))
                st.rerun()
        with scale_cols[1]:
            if st.button("➕ 10%", key="scale_up", use_container_width=True):
                st.session_state.box_size = int(st.session_state.box_size * 1.1)
                st.session_state.stride_x = int(st.session_state.stride_x * 1.1)
                st.session_state.stride_y = int(st.session_state.stride_y * 1.1)
                st.rerun()

    with img_col1:
        st.subheader("Original Image")
        
        # Get current settings for box generation
        current_settings = {
            "bbox_origin_grid": st.session_state.bbox_origin_grid,
            "bbox_origin_ca": st.session_state.bbox_origin_ca,
            "box_size": st.session_state.box_size,
            "stride_x": st.session_state.stride_x,
            "stride_y": st.session_state.stride_y,
            "rows": st.session_state.rows,
            "cols": st.session_state.cols,
            "offset_x": st.session_state.offset_x,
            "offset_y": st.session_state.offset_y,
        }
        
        # Generate boxes if anchor is set
        if st.session_state.bbox_origin_grid:
            boxes = generate_boxes(analyzer, current_settings, mode)
            if boxes:
                analyzer.bounding_boxes = boxes
                st.session_state.custom_boxes = boxes
            display_img, scale = resize_for_display(analyzer.draw_boxes(), max_width=600)
        else:
            display_img, scale = resize_for_display(analyzer.slide, max_width=600)
        
        st.session_state.display_scale = scale
        
        coords = streamlit_image_coordinates(display_img, key="img_click")
        
        # Handle click - only process if it's a new click
        if coords:
            click_key = (coords["x"], coords["y"])
            if click_key != st.session_state.last_click_coords:
                st.session_state.last_click_coords = click_key
                orig_x = int(coords["x"] / scale)
                orig_y = int(coords["y"] / scale)
                
                if anchor_mode in ["Grid (A1)"]:
                    st.session_state.bbox_origin_grid = (orig_x, orig_y)
                else:  # CA or Grayscale (GS0)
                    st.session_state.bbox_origin_ca = (orig_x, orig_y)
                st.rerun()

    # ================= UPDATE TRANSFORMED DATA IF BOXES CHANGED =================
    current_box_hash = get_box_settings_hash()
    boxes_changed = current_box_hash != st.session_state.last_box_settings_hash
    st.session_state.last_box_settings_hash = current_box_hash

    if boxes_changed and st.session_state.get("transformed_img") is not None and st.session_state.custom_boxes:
        transformed_path = image_path.replace(".tif", "_transformed.tif")
        st.session_state.transformed_img.save(transformed_path)
        
        if mode == "IT8":
            analyzer_t = IT8Analyzer(transformed_path, ref_path)
        else:
            analyzer_t = AppliedImageAnalyzer(transformed_path)
        
        analyzer_t.rotate(st.session_state.rotate_angle)
        analyzer_t.bounding_boxes = st.session_state.custom_boxes
        st.session_state.df_transformed = pd.DataFrame(analyzer_t.process())

    with img_col2:
        st.subheader("Transformed Image")
        if st.session_state.get("transformed_img") is not None:
            transformed_path = image_path.replace(".tif", "_transformed.tif")
            st.session_state.transformed_img.save(transformed_path)
            
            if mode == "IT8":
                analyzer_t = IT8Analyzer(transformed_path, ref_path)
            else:
                analyzer_t = AppliedImageAnalyzer(transformed_path)
            
            analyzer_t.rotate(st.session_state.rotate_angle)
            
            if st.session_state.custom_boxes:
                analyzer_t.bounding_boxes = st.session_state.custom_boxes
                display_img2, _ = resize_for_display(analyzer_t.draw_boxes(), max_width=600)
            else:
                display_img2, _ = resize_for_display(analyzer_t.slide, max_width=600)
            
            st.image(display_img2)
        else:
            st.info("Click 'Transform' button to generate")

    # ================= TRANSFORM BUTTONS =================
    st.divider()
    
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        transform_icc_clicked = st.button("Transform by ICC", use_container_width=True)
    with btn_col2:
        transform_py_clicked = st.button("Transform by Py", use_container_width=True)
    with btn_col3:
        if st.session_state.get("transformed_img"):
            buf = io.BytesIO()
            st.session_state.transformed_img.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "generated_image.png", "image/png", use_container_width=True)

    # ================= TRANSFORM LOGIC =================
    transformed_np = None

    if transform_icc_clicked:
        if uploaded_icc is None:
            st.error("Please upload an ICC file.")
            st.stop()
        transformed_np = apply_icc(np.array(analyzer.slide), uploaded_icc.getvalue())

    elif transform_py_clicked:
        if uploaded_py is None:
            st.error("Please upload a Python file.")
            st.stop()
        calibrate_fn = load_py_calibrator(uploaded_py.getvalue())
        transformed_np = calibrate_fn(np.array(analyzer.slide))

    if transformed_np is not None:
        transformed_img = Image.fromarray(transformed_np)
        st.session_state.transformed_img = transformed_img
        
        transformed_path = image_path.replace(".tif", "_transformed.tif")
        transformed_img.save(transformed_path)

        if mode == "IT8":
            analyzer_t = IT8Analyzer(transformed_path, ref_path)
        else:
            analyzer_t = AppliedImageAnalyzer(transformed_path)
        
        analyzer_t.rotate(st.session_state.rotate_angle)
        
        if st.session_state.custom_boxes:
            analyzer_t.bounding_boxes = st.session_state.custom_boxes
        
        st.session_state.df_transformed = pd.DataFrame(analyzer_t.process())
        st.rerun()

    # ================= DATAFRAMES =================
    st.divider()

    if mode != "Apply" and st.session_state.custom_boxes:
        df_col1, df_col2 = st.columns(2)

        analyzer.bounding_boxes = st.session_state.custom_boxes

        with df_col1:
            st.subheader("Original Data")
            df_orig = pd.DataFrame(analyzer.process())
            st.dataframe(df_orig.style.applymap(color_hex_col, subset=['hex']), use_container_width=True, hide_index=True)
            st.metric("Mean ΔE", round(df_orig["delta"].mean(), 2))

        with df_col2:
            st.subheader("Transformed Data")
            if st.session_state.get("df_transformed") is not None:
                st.dataframe(st.session_state.df_transformed.style.applymap(color_hex_col, subset=['hex']), use_container_width=True, hide_index=True)
                st.metric("Mean ΔE", round(st.session_state.df_transformed["delta"].mean(), 2))
            else:
                st.info("No transformed data yet")
    elif mode != "Apply":
        st.info("Please click on the image to set the Grid anchor point (A1).")

    # -------- Cleanup --------
    os.unlink(image_path)
    if ref_path:
        os.unlink(ref_path)