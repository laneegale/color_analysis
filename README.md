### Setup & Run

1. Navigate to the project directory:
   ```bash
   cd color_analysis
   ```

2. Install the package in editable mode:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```


### **Calibration Slide Analyzer - User Guide**

This tool allows you to analyze color calibration slides (Applied Image or IT8), align bounding boxes to specific color patches, apply color transformations (ICC or Python-based), and measure color accuracy (Delta E).

---

#### **1. Setup and Uploads**
First, select your **Analysis Mode** at the top:

*   **Applied Image:** For custom color charts (Defaults to 4 rows × 6 columns).
*   **IT8:** For standard IT8 calibration targets (Requires a Reference `.txt` file).
*   **Apply:** For H&E or general image slides.

**Required Uploads:**
*   **Image Slide:** Upload your `.tif`, `.png`, or `.jpg` image.
*   **Reference File:** (IT8 Mode only) Upload the data file provided by the manufacturer.
*   **Transformation Files:** Upload an `.icc` profile or a `.py` script if you intend to correct the colors.

---

#### **2. Aligning the Grid (Crucial Step)**
Once the image loads, you need to tell the system where the color patches are.

1.  **Set the Anchor Point:**
    *   Look at the "Original Image" display.
    *   Select **"Grid (A1)"** in the radio buttons below the image.
    *   **Click on the center of the first color patch** (Top-Left corner, usually "A1").
    *   *Optional:* If using a secondary anchor (like a Gray Scale strip), select "GS" or "CA" and click that patch.

2.  **Fine-Tuning the Boxes:**
    *   **Move:** Use the Arrow buttons (▲ ▼ ◀ ▶) in the "Controls" column to nudge the entire grid. Change the "Step (px)" value for faster or finer movement.
    *   **Rotate:** Use the **Rotate Slider** to align the grid if your scanned image is crooked.
    *   **Resize:** Use **Scale Boxes (+ / -)** to shrink or grow the detection squares.
    *   **Grid Settings:** Expand the **"Bounding Box Generator"** section to manually change the number of Rows/Columns or the spacing (Stride) if the defaults don't match your image.

---

#### **3. Applying Transformations**
You can correct the colors of your original image using one of two methods:

*   **Transform by ICC:**
    *   Ensure you have uploaded an `.icc` or `.icm` file.
    *   Click the **"Transform by ICC"** button.
*   **Transform by Py:**
    *   Ensure you have uploaded a Python script.
    *   **Requirement:** The script must contain a function named `calibrate(img_np)` that accepts a numpy array and returns a corrected numpy array.
    *   Click the **"Transform by Py"** button.

The app will generate a "Transformed Image" on the right side. The bounding boxes will automatically sync to match the original image.

---

#### **4. Analyzing Results**
Scroll down to the bottom to see the data:

*   **Original Data:** Shows the read color values (Hex) and Delta E (error) compared to the reference.
*   **Transformed Data:** Shows the results after your transformation is applied.
*   **Mean ΔE:** Look at this metric. A lower number means better color accuracy (closer to the reference).

**Exporting:**
*   Click **"Download Transformed"** to save the corrected image as a PNG.

---

#### **💡 Python Script Example**
If you are uploading a `.py` file for transformation, it must look like this:

```python
import numpy as np

def calibrate(img_np):
    # Your custom logic here
    # Example: Simple brightness increase
    return np.clip(img_np * 1.1, 0, 255).astype(np.uint8)
```