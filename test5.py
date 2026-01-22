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

MODE_DEFAULTS = {
    "Applied Image": {"box_size": 12, "stride_x": 55, "stride_y": 44, "rows": 4, "cols": 6},
    "IT8": {"box_size": 20, "stride_x": 40, "stride_y": 40, "rows": 12, "cols": 22},
    "Apply": {"box_size": 12, "stride_x": 55, "stride_y": 44, "rows": 4, "cols": 6},
}

# ================== STYLING ==================
st.markdown(
    """
<style>
    [data-testid="stImage"] { display: flex; justify-content: center; }
    .stButton > button { border-radius: 8px; font-size: 1.2rem; font-weight: bold; }
    div[data-testid="column"] .stButton > button { padding: 0.3rem 0.5rem; min-height: 2.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ================== HELPER FUNCTIONS ==================
def create_analyzer(image_path, ref_path, mode):
    """Factory function to create the appropriate analyzer"""
    if mode == "IT8":
        return IT8Analyzer(image_path, ref_path)
    return AppliedImageAnalyzer(image_path)


def resize_for_display(img: Image.Image, max_width=800):
    w, h = img.size
    if w <= max_width:
        return img, 1.0
    scale = max_width / w
    return img.resize((max_width, int(h * scale)), Image.BICUBIC), scale


def apply_icc_transform(img_np, icc_bytes):
    img = Image.fromarray(img_np)
    with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
        f.write(icc_bytes)
        icc_path = f.name
    try:
        srgb = ImageCms.createProfile("sRGB")
        src_profile = ImageCms.ImageCmsProfile(icc_path)
        img_icc = ImageCms.profileToProfile(img, src_profile, srgb, outputMode=img.mode)
        return np.array(img_icc)
    finally:
        os.unlink(icc_path)


def load_py_calibrator(py_bytes):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(py_bytes)
        py_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("user_calib", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "calibrate"):
            raise ValueError("Python file must define a calibrate(img_np) function")
        return module.calibrate
    finally:
        os.unlink(py_path)


def generate_bounding_boxes(analyzer):
    """Generate bounding boxes using current session state settings"""
    if st.session_state.bbox_origin_grid is None:
        return None

    origin = (
        st.session_state.bbox_origin_grid[0] + st.session_state.offset_x,
        st.session_state.bbox_origin_grid[1] + st.session_state.offset_y,
    )

    ca_origin = None
    if st.session_state.bbox_origin_ca:
        ca_origin = (
            st.session_state.bbox_origin_ca[0] + st.session_state.offset_x,
            st.session_state.bbox_origin_ca[1] + st.session_state.offset_y,
        )

    return analyzer._generate_bounding_boxes(
        origin=origin,
        rows=st.session_state.rows,
        cols=st.session_state.cols,
        box_w=st.session_state.box_size,
        box_h=st.session_state.box_size,
        stride_x=st.session_state.stride_x,
        stride_y=st.session_state.stride_y,
        anchor="center",
        ca_origin=ca_origin,
    )


def sync_analyzers():
    """Sync rotation and bounding boxes between original and transformed analyzers"""
    angle = st.session_state.rotate_angle

    # Update original analyzer
    if st.session_state.analyzer_original:
        st.session_state.analyzer_original.rotate(angle)
        boxes = generate_bounding_boxes(st.session_state.analyzer_original)
        if boxes:
            st.session_state.analyzer_original.bounding_boxes = boxes
            st.session_state.current_boxes = boxes

    # Update transformed analyzer with same settings
    if st.session_state.analyzer_transformed:
        st.session_state.analyzer_transformed.rotate(angle)
        if st.session_state.current_boxes:
            st.session_state.analyzer_transformed.bounding_boxes = st.session_state.current_boxes


def apply_transform(transform_fn):
    """Apply transformation to original image and create transformed analyzer"""
    if not st.session_state.analyzer_original:
        return False

    # Get the BASE (unrotated) image, apply transform, then we'll rotate the analyzer
    base_img = np.array(st.session_state.analyzer_original.base_slide)
    transformed_np = transform_fn(base_img)
    transformed_img = Image.fromarray(transformed_np)

    # Save to temp file and create new analyzer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
        transformed_img.save(f.name)
        temp_path = f.name

    try:
        st.session_state.analyzer_transformed = create_analyzer(
            temp_path, st.session_state.ref_path, st.session_state.mode
        )
        st.session_state.transformed_img = transformed_img

        # Sync rotation and boxes
        sync_analyzers()
        return True
    finally:
        os.unlink(temp_path)


def get_settings_hash():
    """Hash of settings that affect box generation"""
    return (
        st.session_state.box_size,
        st.session_state.stride_x,
        st.session_state.stride_y,
        st.session_state.rows,
        st.session_state.cols,
        st.session_state.rotate_angle,
        st.session_state.offset_x,
        st.session_state.offset_y,
        st.session_state.bbox_origin_grid,
        st.session_state.bbox_origin_ca,
    )


def reset_state_for_new_file():
    """Reset relevant state when a new file is loaded"""
    st.session_state.analyzer_original = None
    st.session_state.analyzer_transformed = None
    st.session_state.transformed_img = None
    st.session_state.current_boxes = None
    st.session_state.bbox_origin_grid = None
    st.session_state.bbox_origin_ca = None
    st.session_state.offset_x = 0
    st.session_state.offset_y = 0
    st.session_state.rotate_angle = 0.0  # ensure Rotate slider starts at 0 for each new file
    st.session_state.last_click_coords = None
    st.session_state.settings_hash = None


# ================== SESSION STATE INITIALIZATION ==================
def init_session_state():
    defaults = {
        # Analyzers
        "analyzer_original": None,
        "analyzer_transformed": None,
        "transformed_img": None,
        "current_boxes": None,
        # File tracking
        "image_path": None,
        "ref_path": None,
        "mode": None,
        "file_key": None,
        # Box settings
        "bbox_origin_grid": None,
        "bbox_origin_ca": None,
        "box_size": 12,
        "stride_x": 55,
        "stride_y": 44,
        "rows": 4,
        "cols": 6,
        "rotate_angle": 0.0,
        "offset_x": 0,
        "offset_y": 0,
        # UI state
        "display_scale": 1.0,
        "last_click_coords": None,
        "anchor_mode": "Grid (A1)",
        "settings_hash": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ================== MAIN APP ==================
st.set_page_config(layout="wide")
st.title("Calibration Slide Analyzer")

init_session_state()

# -------- Mode Selection --------
mode = st.radio("Analysis Mode", options=["Applied Image", "IT8", "Apply"], horizontal=True)

# Update defaults when mode changes
if st.session_state.mode != mode:
    st.session_state.mode = mode
    mode_defaults = MODE_DEFAULTS[mode]
    for key, val in mode_defaults.items():
        st.session_state[key] = val
    st.session_state.rotate_angle = 0.0  # ensure Rotate starts at 0 when mode changes

# -------- File Uploads --------
col1, col2, col3, col4 = st.columns(4)

with col1:
    labels = {"Applied Image": "Applied Image Slide", "IT8": "IT8 Image Slide", "Apply": "H&E Image Slide"}
    uploaded_img = st.file_uploader(f"Upload {labels[mode]} (.tif)", type=IMAGE_TYPES)

with col2:
    uploaded_ref = st.file_uploader("Upload IT8 Reference File (.txt)", type=["txt"]) if mode == "IT8" else None

with col3:
    uploaded_icc = st.file_uploader("ICC file", type=["icm", "icc"])

with col4:
    uploaded_py = st.file_uploader("Py transform file", type=["py"], help="Must contain 'calibrate(img_np)' function")

# -------- Check if ready --------
ready = uploaded_img and (mode in ["Applied Image", "Apply"] or uploaded_ref)

if not ready:
    st.info("Please upload the required files to continue.")
    st.stop()

# -------- Initialize/Reset Analyzers on File Change --------
file_key = (uploaded_img.name, uploaded_ref.name if uploaded_ref else None, mode)

if st.session_state.file_key != file_key:
    reset_state_for_new_file()
    st.session_state.file_key = file_key

    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        uploaded_img.seek(0)
        tmp.write(uploaded_img.read())
        st.session_state.image_path = tmp.name

    if uploaded_ref:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            uploaded_ref.seek(0)
            tmp.write(uploaded_ref.read())
            st.session_state.ref_path = tmp.name
    else:
        st.session_state.ref_path = None

    # Create original analyzer
    st.session_state.analyzer_original = create_analyzer(
        st.session_state.image_path, st.session_state.ref_path, mode
    )

# -------- Bounding Box Settings --------
with st.expander("Bounding Box Generator", expanded=True):
    if mode == "IT8":
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

    st.slider(
        "Rotate",
        -10.0,
        10.0,
        value=0.0,   # initial value (used when session_state doesn't already have rotate_angle)
        step=0.1,
        key="rotate_angle",
    )

# -------- Sync analyzers when settings change --------
current_hash = get_settings_hash()
if current_hash != st.session_state.settings_hash:
    st.session_state.settings_hash = current_hash
    sync_analyzers()

# -------- Image Display Row --------
margin1, img_col1, ctrl_col, img_col2, margin2 = st.columns([0.5, 4, 1, 4, 0.5])

with img_col1:
    st.subheader("Original Image")

    analyzer = st.session_state.analyzer_original
    if analyzer:
        if st.session_state.current_boxes:
            display_img, scale = resize_for_display(analyzer.draw_boxes(), max_width=800)
        else:
            display_img, scale = resize_for_display(analyzer.slide, max_width=800)

        st.session_state.display_scale = scale
        coords = streamlit_image_coordinates(display_img, key="img_click")

        # Handle click
        if coords:
            click_key = (coords["x"], coords["y"])
            if click_key != st.session_state.last_click_coords:
                st.session_state.last_click_coords = click_key
                orig_x = int(coords["x"] / scale)
                orig_y = int(coords["y"] / scale)

                if st.session_state.anchor_mode == "Grid (A1)":
                    st.session_state.bbox_origin_grid = (orig_x, orig_y)
                else:
                    st.session_state.bbox_origin_ca = (orig_x, orig_y)
                st.rerun()

    # Anchor controls
    st.markdown("---")
    anchor_col1, anchor_col2 = st.columns(2)

    with anchor_col1:
        options = ["Grid (A1)", "Grayscale (GS0)"] if mode == "IT8" else ["Grid (A1)", "CA"]
        st.radio(
            "Click image to set:",
            options,
            horizontal=True,
            key="anchor_mode",
        )

    with anchor_col2:
        s1, s2 = st.columns(2)
        with s1:
            if st.session_state.bbox_origin_grid:
                st.success(f"Grid: {st.session_state.bbox_origin_grid}")
            else:
                st.warning("Grid not set")
        with s2:
            label = "GS" if mode == "IT8" else "CA"
            if st.session_state.bbox_origin_ca:
                st.success(f"{label}: {st.session_state.bbox_origin_ca}")
            else:
                st.caption(f"{label} not set")

with ctrl_col:
    st.subheader("Controls")

    st.markdown("##### Move Boxes")
    move_step = st.number_input("Step (px)", 1, 100, 5, key="move_step")

    # Arrow buttons
    _, up_col, _ = st.columns([1, 1, 1])
    with up_col:
        if st.button("▲", key="up"):
            st.session_state.offset_y -= move_step
            st.rerun()

    left_col, _, right_col = st.columns([1, 1, 1])
    with left_col:
        if st.button("◀", key="left"):
            st.session_state.offset_x -= move_step
            st.rerun()
    with right_col:
        if st.button("▶", key="right"):
            st.session_state.offset_x += move_step
            st.rerun()

    _, down_col, _ = st.columns([1, 1, 1])
    with down_col:
        if st.button("▼", key="down"):
            st.session_state.offset_y += move_step
            st.rerun()

    st.markdown("##### Scale Boxes")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("−", key="scale_down"):
            st.session_state.box_size = max(1, int(st.session_state.box_size * 0.9))
            st.session_state.stride_x = max(1, int(st.session_state.stride_x * 0.9))
            st.session_state.stride_y = max(1, int(st.session_state.stride_y * 0.9))
            st.rerun()
    with sc2:
        # Use a full-width plus to avoid font/rendering issues while keeping the UI/functionality the same.
        if st.button("＋", key="scale_up"):
            st.session_state.box_size = int(st.session_state.box_size * 1.1)
            st.session_state.stride_x = int(st.session_state.stride_x * 1.1)
            st.session_state.stride_y = int(st.session_state.stride_y * 1.1)
            st.rerun()

with img_col2:
    st.subheader("Transformed Image")

    analyzer_t = st.session_state.analyzer_transformed
    if analyzer_t:
        if st.session_state.current_boxes:
            display_img2, _ = resize_for_display(analyzer_t.draw_boxes(), max_width=800)
        else:
            display_img2, _ = resize_for_display(analyzer_t.slide, max_width=800)
        st.image(display_img2)
    else:
        st.info("Click 'Transform' button to generate")

# -------- Transform Buttons --------
st.divider()

btn1, btn2, btn3, _ = st.columns(4)

with btn1:
    if st.button("Transform by ICC"):
        if not uploaded_icc:
            st.error("Please upload an ICC file.")
        else:
            transform_fn = lambda img: apply_icc_transform(img, uploaded_icc.getvalue())
            if apply_transform(transform_fn):
                st.rerun()

with btn2:
    if st.button("Transform by Py"):
        if not uploaded_py:
            st.error("Please upload a Python file.")
        else:
            calibrate_fn = load_py_calibrator(uploaded_py.getvalue())
            if apply_transform(calibrate_fn):
                st.rerun()

with btn3:
    if st.session_state.transformed_img:
        buf = io.BytesIO()
        st.session_state.transformed_img.save(buf, format="PNG")
        st.download_button("Download Transformed", buf.getvalue(), "transformed.png", "image/png")

# -------- Data Tables --------
st.divider()

if mode != "Apply" and st.session_state.current_boxes:
    color_hex_col = lambda val: f"background-color: {val};"

    df_col1, df_col2 = st.columns(2)

    with df_col1:
        st.subheader("Original Data")
        df_orig = pd.DataFrame(st.session_state.analyzer_original.process())
        st.dataframe(df_orig.style.map(color_hex_col, subset=["hex"]), hide_index=True)
        st.metric("Mean ΔE", round(df_orig["delta"].mean(), 2))

    with df_col2:
        st.subheader("Transformed Data")
        if st.session_state.analyzer_transformed:
            df_trans = pd.DataFrame(st.session_state.analyzer_transformed.process())
            st.dataframe(df_trans.style.map(color_hex_col, subset=["hex"]), hide_index=True)
            st.metric("Mean ΔE", round(df_trans["delta"].mean(), 2))
        else:
            st.info("No transformed data yet")
elif mode != "Apply":
    st.info("Please click on the image to set the Grid anchor point (A1).")