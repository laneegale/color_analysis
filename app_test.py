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

def generate_grid_boxes(origin, rows, cols, box_w, box_h, stride_x, stride_y, anchor="top-left", label_prefix="B"):
    boxes = {}
    ox, oy = origin
    if anchor == "center":
        ox -= box_w // 2
        oy -= box_h // 2
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(ox + c * stride_x), int(oy + r * stride_y)
            x2, y2 = x1 + box_w, y1 + box_h
            boxes[f"{label_prefix}{idx}"] = (y1, y2, x1, x2)
            idx += 1
    return boxes

def resize_for_display(img: Image.Image, max_width=600):
    w, h = img.size
    if w <= max_width:
        return img
    scale = max_width / w
    return img.resize((max_width, int(h * scale)), Image.BICUBIC)

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

st.set_page_config(layout="wide")
st.title("Calibration Slide Analyzer")

if "bbox_origin" not in st.session_state:
    st.session_state.bbox_origin = None
if "custom_boxes" not in st.session_state:
    st.session_state.custom_boxes = None

# -------- Mode Toggle --------
mode = st.radio("Analysis Mode", options=["Applied Image", "IT8", "Apply"], horizontal=True)

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
        AnalyzerClass, analyzer_args = IT8Analyzer, (image_path, ref_path)
    else:
        analyzer = AppliedImageAnalyzer(image_path)
        AnalyzerClass, analyzer_args = AppliedImageAnalyzer, (image_path,)

    color_hex_col = lambda val: f"background-color: {val};"

    # ================= ROW 1: IMAGES =================
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.subheader("Original Image")
        if mode in ["Applied Image", "IT8"]:
            angle = st.session_state.get("rotate_angle", 0.0)
            analyzer.rotate(angle)
            display_img = resize_for_display(analyzer.slide, max_width=600)
            # rotated_img = base_img.rotate(angle, resample=Image.BICUBIC, expand=False)
            # analyzer.rotated_img = rotated_img
            coords = streamlit_image_coordinates(display_img)
            if coords:
                st.session_state.bbox_origin = (int(coords["x"]), int(coords["y"]))
            if st.session_state.bbox_origin:
                st.success(f"Anchor point: {st.session_state.bbox_origin}")
        else:
            display_img = resize_for_display(analyzer.draw_boxes(), max_width=600)
            st.image(display_img)

    with img_col2:
        st.subheader("Transformed Image")
        if st.session_state.get("transformed_img") is not None:
            base_img2 = resize_for_display(st.session_state.transformed_img, max_width=600)
            angle = st.session_state.get("rotate_angle", 0.0)
            rotated_img2 = base_img2.rotate(angle, resample=Image.BICUBIC, expand=False)
            st.session_state.transformed_img_rotated = rotated_img2
            coords = streamlit_image_coordinates(rotated_img2)
            # st.image(display_img2)
        else:
            st.info("Click 'Transform' button to generate")

    # ================= ROW 2: BOUNDING BOX GENERATOR =================
    st.divider()

    with st.expander("Bounding Box Generator", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        box_w = c1.number_input("Box W", 10, 500, 64)
        box_h = c2.number_input("Box H", 10, 500, 64)
        stride_x = c3.number_input("Stride X", 10, 500, 72)
        stride_y = c4.number_input("Stride Y", 10, 500, 72)

        c5, c6, c7, c8 = st.columns(4)
        rows = c5.number_input("Rows", 1, 50, 12)
        cols = c6.number_input("Cols", 1, 50, 22)
        anchor = c7.selectbox("Anchor", ["top-left", "center"])
        angle = c8.slider("Rotate", -10.0, 10.0, 0.0, 0.1, key="rotate_angle")
        # angle = c8.number_input("Rotate", -180.0, 180.0, 0.0, 0.1, key="rotate_angle")

        if st.button("Apply Bounding Boxes"):
            if st.session_state.bbox_origin is None:
                st.error("Please click on the image to set an anchor point.")
            else:
                boxes = generate_grid_boxes(
                    st.session_state.bbox_origin, rows, cols,
                    box_w, box_h, stride_x, stride_y, anchor
                )
                analyzer.bounding_boxes = st.session_state.custom_boxes = boxes

    # ================= ROW 3: TRANSFORM BUTTONS =================
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        transform_icc_clicked = st.button("Transform by ICC", use_container_width=True)
    with btn_col2:
        transform_py_clicked = st.button("Transform by Py", use_container_width=True)
    with btn_col3:
        if st.session_state.get("transformed_img") :
            buf = io.BytesIO()
            if st.session_state.get("transformed_img_rotated"):
                st.session_state.transformed_img_rotated.save(buf, format="PNG")
            else:
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
        transformed_path = image_path.replace(".tif", "_transformed.tif")
        transformed_img.save(transformed_path)

        analyzer_t = AnalyzerClass(transformed_path, ref_path) if mode == "IT8" else AnalyzerClass(transformed_path)

        st.session_state.transformed_img = transformed_img
        st.session_state.df_transformed = pd.DataFrame(analyzer_t.process())
        st.rerun()

    # ================= ROW 4: DATAFRAMES =================
    st.divider()

    if mode != "Apply":
        df_col1, df_col2 = st.columns(2)

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

    # -------- Cleanup --------
    os.unlink(image_path)
    if ref_path:
        os.unlink(ref_path)