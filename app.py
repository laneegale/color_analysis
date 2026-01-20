import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageCms
import tempfile
import os
import io
import importlib.util

from src.calibration_analyzer.analyzer import AppliedImageAnalyzer, IT8Analyzer
from src.calibration_analyzer.CEL_GOG_Matrix_Conversion_1113 import transform_img

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

    img_icc = ImageCms.profileToProfile(
        img, src_profile, srgb, outputMode=img.mode
    )
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

# -------- Mode Toggle --------
mode = st.radio(
    "Analysis Mode",
    options=["Applied Image", "IT8", "Apply"],
    horizontal=True
)

# -------- File Uploads --------
if mode == "Applied Image":
    uploaded_img = st.file_uploader(
        "Upload Applied Image Slide (.tif)",
        type=["tif", "tiff"]
    )
    uploaded_ref = None
elif mode == "IT8":
    uploaded_img = st.file_uploader(
        "Upload IT8 Image Slide (.tif)",
        type=["tif", "tiff"]
    )
    uploaded_ref = st.file_uploader(
        "Upload IT8 Reference File (.txt)",
        type=["txt"]
    )
else:
    uploaded_img = st.file_uploader(
        "Upload H&E Image Slide (.tif)",
        type=["tif", "tiff"]
    )
    uploaded_ref = None

uploaded_icc = st.file_uploader(
    "ICC file (Optional)",
    type=["icm", "icc"]
)
uploaded_py = st.file_uploader(
    "py transform file (Optional, make sure in this file there is a function named 'calibrate' where both input and output is np array)",
    type=["py"]
)
# -------- Proceed only if required files exist --------
ready = uploaded_img and (mode == "Applied Image" or mode == "Apply" or uploaded_ref)

if ready:
    # Reset state on mode/file change
    state_key = (
        uploaded_img.name,
        uploaded_ref.name if uploaded_ref else None,
        mode
    )

    if st.session_state.get("last_state") != state_key:
        st.session_state.transformed_img = None
        st.session_state.df_transformed = None
        st.session_state.last_state = state_key

    # -------- Save image file --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_img:
        tmp_img.write(uploaded_img.read())
        image_path = tmp_img.name

    # -------- Save reference file if IT8 --------
    ref_path = None
    if uploaded_ref:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_ref:
            tmp_ref.write(uploaded_ref.read())
            ref_path = tmp_ref.name

    # -------- Analyzer selection --------
    if mode == "Applied Image":
        analyzer = AppliedImageAnalyzer(image_path)
        AnalyzerClass = AppliedImageAnalyzer
        analyzer_args = (image_path,)
    elif mode == "IT8":
        analyzer = IT8Analyzer(image_path, ref_path)
        AnalyzerClass = IT8Analyzer
        analyzer_args = (image_path, ref_path)
    else:
        analyzer = AppliedImageAnalyzer(image_path)
        AnalyzerClass = AppliedImageAnalyzer
        analyzer_args = (image_path,)

    # -------- Transform Button (TOP) --------
    bt1, bt2, bt3 = st.columns([0.2, 0.2, 1])
    with bt1:
        transform_default_clicked = st.button("Transform Image (Default)")
    with bt2:
        transform_icc_clicked = st.button("Transform by ICC")
    with bt3:
        transform_py_clicked = st.button("Transform by py")

    # -------- Layout --------
    col1, col2 = st.columns(2)

    # ================= LEFT: ORIGINAL =================
    with col1:
        st.subheader("Original Image")
        if mode == "Apply":
            display_img = resize_for_display(analyzer.slide, max_width=600)
            st.image(display_img)
        else:
            display_img = resize_for_display(analyzer.draw_boxes(), max_width=600)
            st.image(display_img)

            df_orig = pd.DataFrame(analyzer.process())
            st.dataframe(df_orig, use_container_width=True, hide_index=True)
            st.metric("Mean ΔE", round(df_orig["delta"].mean(), 2))

    # ================= RIGHT: TRANSFORMED =================
    with col2:
        st.subheader("Transformed Image")

        transformed_np = None 
        if transform_default_clicked:
            # --- YOUR EXACT TRANSFORM LOGIC ---
            img_np = np.array(analyzer.slide)
            transformed_np = transform_img(img_np)

        elif transform_icc_clicked:
            if uploaded_icc is None:
                st.error("Please upload an ICC file.")
                st.stop()

            img_np = np.array(analyzer.slide)
            transformed_np = apply_icc(img_np, uploaded_icc.getvalue())

        elif transform_py_clicked:
            if uploaded_py is None:
                st.error("Please upload a Python file.")
                st.stop()

            img_np = np.array(analyzer.slide)
            calibrate_fn = load_py_calibrator(uploaded_py.getvalue())
            transformed_np = calibrate_fn(img_np)

        if transformed_np is not None:
            transformed_img = Image.fromarray(transformed_np)

            transformed_path = image_path.replace(".tif", "_transformed.tif")
            transformed_img.save(transformed_path)

            analyzer_t = AnalyzerClass(*analyzer_args[:-1], transformed_path) \
                if mode in ["Applied Image", "Apply"]  \
                else AnalyzerClass(transformed_path, ref_path)

            st.session_state.transformed_img = transformed_img
            st.session_state.df_transformed = pd.DataFrame(
                analyzer_t.process()
            )


        if st.session_state.get("transformed_img") is not None:
            display_img2 = resize_for_display(st.session_state.transformed_img, max_width=600)
            st.image(display_img2)
            if mode != "Apply":
                st.dataframe(
                    st.session_state.df_transformed,
                    use_container_width=True,
                    hide_index=True
                )
            buf = io.BytesIO()
            st.session_state.transformed_img.save(buf, format="PNG") 
            byte_im = buf.getvalue()
            if mode != "Apply":
                st.metric(
                    "Mean ΔE",
                    round(st.session_state.df_transformed["delta"].mean(), 2)
                )

            st.download_button(
                label="Download Generated Image",
                data=byte_im,
                file_name="generated_image.png",
                mime="image/png",
            )
        else:
            st.info("Click 'Transform Image' to generate")

    # -------- Cleanup --------
    os.unlink(image_path)
    if ref_path:
        os.unlink(ref_path)