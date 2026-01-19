from PIL import Image, ImageFile, ImageCms, PngImagePlugin
import numpy as np
import pandas as pd
from skimage import color
import os
from skimage.color import lab2rgb

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# reference file path
file_path = r"d:\KennyD\SilverFast file\IT8 Reference File SilverFast E130102.csv".replace("\\", "/")
# file_path = r"d:\KennyD\SilverFast file\IT8 Reference File SilverFast E220714.csv".replace("\\", "/")

df = pd.read_csv(file_path)
df = df[["SAMPLE_ID", "LAB_L", "LAB_A", "LAB_B"]]


def get_lab_color(sample_id):
    sample = df[df["SAMPLE_ID"] == sample_id]
    lab = sample[["LAB_L", "LAB_A", "LAB_B"]].values[0]
    return lab

def average_variances(image):
    image = np.array(image)
    image_flat = image.reshape(-1, 3)
    
    image_mean = np.mean(image_flat, axis=0)
    centered_image = image_flat - image_mean
    covariance_matrix = np.cov(centered_image, rowvar=False)
    variances = np.diag(covariance_matrix)
    average_variances = variances.sum() / len(variances)

    return image_mean.astype(int), average_variances

def cie2000(image_mean, default_lab):
    image_mean_lab = color.rgb2lab(np.array(image_mean) / 255)
    cie = color.deltaE_ciede2000(image_mean_lab, default_lab)
    return image_mean_lab, cie


num_rows = 12
num_cols = 22
char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

sample_id_list = []
rgb_r = []
rgb_g = []
rgb_b = []
lab_l = []
lab_a = []
lab_b = []
variance = []
cie_delta_2000 = []

# 需要測量6個數字 請看it8_guideline
# 1. A11 A12上方的中間位置
# 2. F1 G1左方的中間位置
# 3. 顏色格的平均大小
# 4. 下方白色格0號位置的左上位置(y coordinate) 請預留少許空間
# 5. 下方白色格0號位置的左上位置(x coordinate)
# 6. 顏色格的平均大小(同3)

# slide input
slide_path = r"x:\KennyX\ICC Data for Adler\KFBIO KF-PRO-005-EX\2025_11_13 Colour Engineering Lab GOG + 3x3 Matrix Conversion\CUHK_ACP_ICC_00005_corrected_sRGB_best_model.tif".replace("\\", "/")
slide = Image.open(slide_path).convert("RGB")
target_icc = None

# target_icc = r"x:\KennyX\ICC Data for Adler\Leica GT450\2025_10_23 00005\CUHK ACP ICC 40x Leica GT450 DX.icm".replace("\\", "/")
# profile_sRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
# profile_target = ImageCms.getOpenProfile(target_icc)
# slide = ImageCms.profileToProfile(slide, profile_target, profile_sRGB)

slide_np = np.array(slide)

directory = os.path.dirname(slide_path)
if target_icc:
    file_name = os.path.splitext(os.path.basename(target_icc))[0]
else:
    file_name = os.path.splitext(os.path.basename(slide_path))[0]

# variable setup
input_list = [70, 67, 52, 779, 27, 52]
al_start_top = input_list[0]
al_start_left = input_list[1]
al_square_length = input_list[2]
al_roi_length = round(al_square_length * 0.2)
gs_start_top = input_list[3]
gs_start_left = input_list[4]
gs_square_length = input_list[5]
gs_roi_length = round(gs_square_length * 0.2)


al_tile_np = slide_np[al_start_top: , al_start_left:]  # Remove alpha channel if present
# al_tile = Image.fromarray(al_tile_np)

gs_tile_np = slide_np[gs_start_top: , gs_start_left:]  # Remove alpha channel if present
# gs_tile = Image.fromarray(gs_tile_np)

# al_tile.show()
# gs_tile.show()

# for A to L
for row in range(num_rows):
    for col in range(num_cols):
        sample_id = char_list[row] + str(col + 1)
        lab = get_lab_color(sample_id)

        top = al_square_length * row + (al_square_length - al_roi_length) // 2
        bottom = top + al_roi_length
        left = al_square_length * col + (al_square_length - al_roi_length) // 2
        right = left + al_roi_length
        tile = al_tile_np[top:bottom, left:right]

        # if col == 21:
        #     tile_img = Image.fromarray(tile)
        #     tile_img.show()

        mean, variances = average_variances(tile)
        # print(mean, variances)
        mean_lab, cie = cie2000(mean, lab)
        # print(sample_id, mean_lab, lab, cie)

        sample_id_list.append(sample_id)
        rgb_r.append(mean[0])
        rgb_g.append(mean[1])
        rgb_b.append(mean[2])
        lab_l.append(round(mean_lab[0], 2))
        lab_a.append(round(mean_lab[1], 2))
        lab_b.append(round(mean_lab[2], 2))
        variance.append(round(variances, 1))
        cie_delta_2000.append(round(cie, 2))

# for GS
for i in range(24):
    sample_id = "GS" + str(i)
    lab = get_lab_color(sample_id)

    top = (gs_square_length - gs_roi_length) // 2
    bottom = top + gs_roi_length
    left = gs_square_length * i + (gs_square_length - gs_roi_length) // 2
    right = left + gs_roi_length
    tile = gs_tile_np[top:bottom, left:right]

    # # if i == 23:
    # if i in [22, 23]:
    #     tile_img = Image.fromarray(tile)
    #     tile_img.show()

    mean, variances = average_variances(tile)
    # print(mean, variances)
    mean_lab, cie = cie2000(mean, lab)
    # print(sample_id, mean_lab, lab, cie)

    sample_id_list.append(sample_id)
    rgb_r.append(mean[0])
    rgb_g.append(mean[1])
    rgb_b.append(mean[2])
    lab_l.append(round(mean_lab[0], 2))
    lab_a.append(round(mean_lab[1], 2))
    lab_b.append(round(mean_lab[2], 2))
    variance.append(round(variances, 1))
    cie_delta_2000.append(round(cie, 2))

color_df = pd.DataFrame(index=range(len(sample_id_list)), columns=["sample_id", "rgb_r", "rgb_g", "rgb_b", "lab_l", "lab_a", "lab_b", "variance", "cie_delta_2000"])

color_df["sample_id"] = sample_id_list
color_df["rgb_r"] = rgb_r
color_df["rgb_g"] = rgb_g
color_df["rgb_b"] = rgb_b
color_df["lab_l"] = lab_l
color_df["lab_a"] = lab_a
color_df["lab_b"] = lab_b
color_df["variance"] = variance
color_df["cie_delta_2000"] = cie_delta_2000


csv_name = file_name + ".csv"
save_folder_path = os.path.join(directory, csv_name)
color_df.to_csv(save_folder_path, index=False)

