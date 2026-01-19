import numpy as np

def transform_img(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Using best model to correct colors of input RGB image with uint8 format.
    best model checkpoint: 1113, trained on IT8 (264 patches) using GOG model.
    Input:
        img_rgb_uint8: (H, W, 3), uint8, [0,255] device RGB
    Output:
        corrected_uint8: (H, W, 3), uint8, [0,255] sRGB
    """
    if img_rgb_uint8.dtype != np.uint8:
        raise ValueError("Input image must be uint8.")
    if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[2] != 3:
        raise ValueError("Input image must have shape (H, W, 3).")

    # Convert to [0,1] float32
    img = img_rgb_uint8.astype(np.float32) / 255.0
    h, w, _ = img.shape
    rgb = img.reshape(-1, 3)

    # Internalized parameters
    A = np.array(
        [
            [6.51679838e-01, 2.09356982e-01, -5.12886103e-02],
            [1.87130686e-01, 8.32886791e-01, -2.73079649e-01],
            [6.50650086e-02, -9.60025506e-02, 1.39492913e00],
        ],
        dtype=np.float32,
    )
    gammas = np.array(
        [1.37733236e00, 1.36402304e00, 1.37560301e00],
        dtype=np.float32,
    )
    k = np.array(
        [1.00581713e00, 1.00756712e00, 1.01099317e00],
        dtype=np.float32,
    )
    b = np.array(
        [-5.99231643e-03, 8.91712312e-04, -2.50402038e-03],
        dtype=np.float32,
    )

    # Model: clip -> per-channel gamma -> *k + b -> 3x3 matrix -> XYZ -> sRGB
    rgb_clipped = np.clip(rgb, 0.0, 1.0)
    rgb_pow = np.power(rgb_clipped, gammas)
    rgb_lin = rgb_pow * k + b
    xyz = rgb_lin @ A

    matrix_xyz_to_srgb = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    srgb_linear = xyz @ matrix_xyz_to_srgb.T
    # Clip to [0,1], then convert back to uint8
    srgb_linear = np.clip(srgb_linear, 0.0, 1.0)
    # apply gamma correction for sRGB
    srgb = np.where(
        srgb_linear <= 0.0031308,
        srgb_linear * 12.92,
        1.055 * np.power(srgb_linear, 1.0 / 2.4) - 0.055,
    )
    corrected_uint8 = (srgb.reshape(h, w, 3) * 255.0 + 0.5).astype(np.uint8)
    return corrected_uint8


# def main():
#     # test on a sample image
#     with tifffile.TiffFile(
#         "IT8 E130102.tif"
#     ) as tif:
#         page = tif.pages[0]
#         it8 = page.asarray()[..., :-1]  # get (H,W,3) uint8

#     corrected = apply_model_rgb_uint8(it8)

#     # save result
#     tifffile.imwrite(
#         "IT8 E130102_icc.tif",
#         corrected,
#         photometric="rgb",
#     )


# if __name__ == "__main__":
#     main()
