from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000

class BaseImageAnalyzer(ABC):
    """
    Abstract base class for calibration slide ROI analyzers.
    """

    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.base_slide = Image.open(slide_path).convert("RGB")
        self.base_image_np = np.asarray(self.base_slide)
        self.slide = Image.open(slide_path).convert("RGB")
        self.image_np = np.asarray(self.slide)
        self.h, self.w = self.image_np.shape[:2]

        # self.bounding_boxes = self._generate_bounding_boxes()
        self.bounding_boxes = []
        self.labels = None

    @abstractmethod
    def _generate_bounding_boxes(self):
        """Return dict: {label: (top, bottom, left, right)}"""
        pass
    
    # @abstractmethod
    # def _generate_icc(self):
    #     pass

    # -------- Shared Functionality --------
    def get_roi(self, label):
        if label not in self.bounding_boxes:
            raise ValueError(f"Label '{label}' not found.")
        t, b, l, r = self.bounding_boxes[label]
        return self.image_np[t:b, l:r]

    def get_avg_color(self, label, round_to=None, to_lab=False):
        roi = self.get_roi(label)
        avg = roi.mean(axis=(0, 1))
        if to_lab:
            return rgb2lab((avg/255)).round(2)
        return avg.round(round_to) if round_to is not None else avg.astype(int)

    def get_variance(self, label, ddof=0):
        roi = self.get_roi(label)
        flat = roi.reshape(-1, 3)
        return flat.var(axis=0, ddof=ddof).mean()

    def get_all_averages(self, round_to=None):
        return {
            label: self.get_avg_color(label, round_to)
            for label in self.bounding_boxes
        }

    def rotate(self, angle):
        self.slide = self.base_slide.rotate(angle, resample=Image.BICUBIC, expand=False)
        self.image_np = np.array(self.slide)

    def draw_boxes(self, color=(0, 255, 0), thickness=2):
        img = self.image_np.copy()
        for t, b, l, r in self.bounding_boxes.values():
            img[t:t+thickness, l:r] = color
            img[b-thickness:b, l:r] = color
            img[t:b, l:l+thickness] = color
            img[t:b, r-thickness:r] = color
        return Image.fromarray(img)

    def compute_delta(self, color1, color2):
        return deltaE_ciede2000(color1, color2)
    
    def process(self):
        if self.labels is None:
            raise Exception("self.labels is None")
        

        def rgb_to_hex(rgb):
            return "#{:02X}{:02X}{:02X}".format(*rgb)
        
        rows = []

        for label in self.labels:
            # --- RGB ---
            rgb = self.get_avg_color(label).astype(int)
            r, g, b = rgb.tolist()

            # --- LAB (measured) ---
            lab = rgb2lab((rgb / 255.0).reshape(1, 1, 3))[0, 0]
            l, a, lab_b = lab.tolist()

            # --- LAB (calibration) ---
            cal_lab = np.array(self.calibration_file[label]["lab"])

            # --- Delta E ---
            delta = self.compute_delta(lab, cal_lab)

            rows.append({
                "label": label,
                "R": r,
                "G": g,
                "B": b,
                "l": round(l, 2),
                "a": round(a, 2),
                "b": round(lab_b, 2),
                "hex": rgb_to_hex(rgb),
                "delta": round(float(delta), 2),
            })
            
        return rows

class AppliedImageAnalyzer(BaseImageAnalyzer):
    def __init__(self, slide_path, num_rows=4, num_cols=6):
        self.num_rows = num_rows
        self.num_cols = num_cols
        rgb_dict = {
            "CA": [205, 212, 210],
            "A1": [0, 106, 147],
            "A2": [146, 35, 39],
            "A3": [160, 61, 115],
            "A4": [26, 126, 65],
            "A5": [224, 161, 0],
            "A6": [0, 57, 118],
            "B1": [69, 60, 86],
            "B2": [205, 211, 201],
            "B3": [174, 180, 171],
            "B4": [136, 148, 142],
            "B5": [162, 59, 71],
            "B6": [194, 98, 12],
            "C1": [210, 137, 0],
            "C2": [147, 160, 35],
            "C3": [97, 111, 106],
            "C4": [61, 77, 72],
            "C5": [32, 46, 43],
            "C6": [24, 79, 139],
            "D1": [24, 164, 153],
            "D2": [101, 117, 150],
            "D3": [75, 98, 59],
            "D4": [49, 109, 135],
            "D5": [175, 129, 104],
            "D6": [95, 73, 58],
        }
        converted = {}
        for key, rgb in rgb_dict.items():
            rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
            rgb_arr = rgb_arr.reshape(1, 1, 3)  # shape required by rgb2lab

            lab = rgb2lab(rgb_arr)[0, 0]

            converted[key] = {
                "rgb": rgb,
                "lab": lab.tolist()
            }
        self.calibration_file = converted

        super().__init__(slide_path)
        self.labels = ["CA"] + [f"{row}{col}" for row in "ABCD" for col in range(1, 7)]


    def _get_config(self):
        r = self.h / 863
        return {
            "ca_w":     round(r * 225),
            "x_margin": round(r * 113),
            "x_gap":    round(r * 81),
            "y_gap":    round(r * 23),
            "diam":     round(r * 200),
            "roi_len":  round(r * 250 * 200 / 4150),
        }

    # def _generate_bounding_boxes(self):
    #     cfg = self._get_config()
    #     size = cfg["roi_len"]
    #     boxes = {}

    #     # Control Area (CA)
    #     ca_top = (self.h // 2) - (size // 2)
    #     ca_left = (cfg["ca_w"] // 2) - (size // 2)
    #     boxes["CA"] = (ca_top, ca_top + size, ca_left, ca_left + size)

    #     # Grid
    #     start_x = cfg["ca_w"] + cfg["x_margin"] + (cfg["diam"] - size) // 2
    #     start_y = (cfg["diam"] - size) // 2
    #     stride_x = cfg["diam"] + cfg["x_gap"]
    #     stride_y = cfg["diam"] + cfg["y_gap"]

    #     row_names = ["A", "B", "C", "D"]

    #     for r in range(self.num_rows):
    #         for c in range(self.num_cols):
    #             top = start_y + stride_y * r
    #             left = start_x + stride_x * c
    #             label = f"{row_names[r]}{c+1}"
    #             boxes[label] = (top, top + size, left, left + size)

    #     return boxes
    
    def _generate_bounding_boxes(
        self,
        origin,
        rows,
        cols,
        box_w,
        box_h,
        stride_x,
        stride_y,
        anchor="top-left",
        ca_origin=None
    ):
        boxes = {}
        ox, oy = origin

        if anchor == "center":
            ox -= box_w // 2
            oy -= box_h // 2

        # Control Area (CA)
        if ca_origin:
            ca_x, ca_y = ca_origin
            if anchor == "center":
                ca_x -= box_w // 2
                ca_y -= box_h // 2
            boxes["CA"] = (ca_y, ca_y + box_h, ca_x, ca_x + box_w)

        # Grid
        row_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]  # Extended for flexibility

        for r in range(rows):
            for c in range(cols):
                x1 = int(ox + c * stride_x)
                y1 = int(oy + r * stride_y)
                x2 = x1 + box_w
                y2 = y1 + box_h

                label = f"{row_names[r % len(row_names)]}{c + 1}"
                boxes[label] = (y1, y2, x1, x2)

        return boxes

class IT8Analyzer(BaseImageAnalyzer):
    def __init__(self, slide_path, ref_file_path, stride=52, offset=21):
        self.stride = stride
        self.offset = offset
        self.roi_size = round(stride * 0.2)

        self.al_origin = (67, 70)
        self.gs_origin = (17, 779)

        self.calibration_file = self._parse_it8_reference_file(ref_file_path)

        super().__init__(slide_path)
        self.labels = [f"{row}{col}" for row in "ABCDEFGHIJKL" for col in range(1, 23)] + [f"GS{col}" for col in range(0, 24)]


    def _parse_it8_reference_file(self, filepath):
        data = {}
        in_format = False
        in_data = False
        columns = []

        with open(filepath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line:
                    continue

                # -----------------------------
                # Parse DATA FORMAT section
                # -----------------------------
                if line == "BEGIN_DATA_FORMAT":
                    in_format = True
                    continue

                if line == "END_DATA_FORMAT":
                    in_format = False
                    continue

                if in_format:
                    columns = line.split()
                    continue

                # -----------------------------
                # Parse DATA section
                # -----------------------------
                if line == "BEGIN_DATA":
                    in_data = True
                    continue

                if line == "END_DATA":
                    break

                if in_data:
                    parts = line.split()
                    row = dict(zip(columns, parts))

                    sample_id = row["SAMPLE_ID"]

                    xyz = (
                        float(row["XYZ_X"]),
                        float(row["XYZ_Y"]),
                        float(row["XYZ_Z"]),
                    )

                    lab = (
                        float(row["LAB_L"]),
                        float(row["LAB_A"]),
                        float(row["LAB_B"]),
                    )

                    data[sample_id] = {
                        "xyz": np.array(xyz),
                        "lab": np.array(lab),
                    }

        return data


    # def _generate_bounding_boxes(self):
    #     boxes = {}

    #     # Main grid A–L, 1–22
    #     row_labels = [chr(i) for i in range(ord('A'), ord('A') + 12)]

    #     start_x = self.al_origin[0] + self.offset
    #     start_y = self.al_origin[1] + self.offset

    #     for r, row in enumerate(row_labels):
    #         for c in range(22):
    #             x = start_x + c * self.stride
    #             y = start_y + r * self.stride
    #             boxes[f"{row}{c+1}"] = (
    #                 y, y + self.roi_size, x, x + self.roi_size
    #             )

    #     # Grayscale strip
    #     gs_x = self.gs_origin[0] + self.offset
    #     gs_y = self.gs_origin[1] + self.offset

    #     for c in range(24):
    #         x = gs_x + c * self.stride
    #         boxes[f"GS{c}"] = (
    #             gs_y, gs_y + self.roi_size, x, x + self.roi_size
    #         )

    #     return boxes
    
    def _generate_bounding_boxes(
        self,
        origin,
        rows=12,  # Fixed for IT8: A-L (ignored, always 12)
        cols=22,  # Fixed for IT8: 1-22 (ignored, always 22)
        box_w=None,
        box_h=None,
        stride_x=None,
        stride_y=None,
        anchor="center",
        ca_origin=None  # For grayscale strip (GS)
    ):
        """
        Generate bounding boxes for IT8 target.
        
        IT8 has fixed structure:
        - Main grid: 12 rows (A-L) x 22 columns (1-22)
        - Grayscale strip: 24 boxes (GS0-GS23)
        """
        boxes = {}
        
        # Use instance defaults if not provided
        box_size = box_w if box_w is not None else self.roi_size
        stride_x = stride_x if stride_x is not None else self.stride
        stride_y = stride_y if stride_y is not None else self.stride
        
        # Main grid A–L, 1–22 (always 12x22 for IT8)
        row_labels = [chr(i) for i in range(ord('A'), ord('A') + 12)]
        
        start_x = origin[0]
        start_y = origin[1]
        
        # Adjust for anchor
        if anchor == "center":
            offset_x = -box_size // 2
            offset_y = -box_size // 2
        else:  # top-left
            offset_x = 0
            offset_y = 0
        
        for r, row in enumerate(row_labels):
            for c in range(22):
                x = start_x + c * stride_x + offset_x
                y = start_y + r * stride_y + offset_y
                boxes[f"{row}{c+1}"] = (
                    y, y + box_size, x, x + box_size
                )
        
        # Grayscale strip (24 boxes)
        if ca_origin:
            gs_x = ca_origin[0]
            gs_y = ca_origin[1]
            
            for c in range(24):
                x = gs_x + c * stride_x + offset_x
                boxes[f"GS{c}"] = (
                    gs_y + offset_y, gs_y + offset_y + box_size, x, x + box_size
                )
        
        return boxes
        