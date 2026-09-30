import os
import numpy as np
from functions import *
from skimage.transform import downscale_local_mean, resize
import rasterio
from rasterio.windows import Window
import math
from scipy.interpolate import Rbf


def read_raster(raster_path):
    dataset = rasterio.open(raster_path)
    raster_profile = dataset.profile
    raster = dataset.read()
    raster = np.transpose(raster, (1, 2, 0))
    raster = raster.astype(np.dtype(raster_profile["dtype"]))

    return raster, raster_profile


def write_raster(raster, raster_profile, raster_path):
    raster_profile["dtype"] = str(raster.dtype)
    raster_profile["height"] = raster.shape[0]
    raster_profile["width"] = raster.shape[1]
    raster_profile["count"] = raster.shape[2]
    image = np.transpose(raster, (2, 0, 1))
    dataset = rasterio.open(raster_path, mode='w', **raster_profile)
    dataset.write(image)
    dataset.close()


def clip_raster(dataset, row_start, row_stop, col_start, col_stop):
    window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
    transform = dataset.window_transform(window)
    clipped_raster = dataset.read(window=window)
    clipped_raster = np.transpose(clipped_raster, (1, 2, 0))
    clipped_profile = dataset.profile
    clipped_profile.update({'width': col_stop - col_start,
                            'height': row_stop - row_start,
                            'transform': transform})

    return clipped_raster, clipped_profile


def color_composite(image, bands_idx):
    image = np.stack([image[:, :, i] for i in bands_idx], axis=2)
    return image


def color_composite_ma(image, bands_idx):
    image = np.ma.stack([image[:, :, i] for i in bands_idx], axis=2)
    return image


def linear_pct_stretch(img, pct=2, max_out=1, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        return gray

    bands = []
    for band_idx in range(img.shape[2]):
        band = img[:, :, band_idx]
        band_strch = gray_process(band)
        bands.append(band_strch)
    img_pct_strch = np.stack(bands, axis=2)
    return img_pct_strch


def linear_pct_stretch_ma(img, pct=2, max_out=1, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        return gray

    out = img.copy()
    for band_idx in range(img.shape[2]):
        band = img.data[:, :, band_idx]
        mask = img.mask[:, :, band_idx]
        band_strch = gray_process(band[~mask])
        out.data[:, :, band_idx][~mask] = band_strch
    return out

def coarse_tps_interpolation(coarse_data, scale_factor, block_size):
    nl, ns, bands = coarse_data.shape
    patch_len = block_size * scale_factor

    n_blocks_row = math.ceil(nl / patch_len)
    n_blocks_col = math.ceil(ns / patch_len)

    result = np.zeros_like(coarse_data, dtype=np.float32)

    for block_row in range(n_blocks_row):
        for block_col in range(n_blocks_col):
            row_start = block_row * patch_len
            row_end = min((block_row + 1) * patch_len, nl)
            col_start = block_col * patch_len
            col_end = min((block_col + 1) * patch_len, ns)

            block_data = coarse_data[row_start:row_end, col_start:col_end, :]

            nl_block = row_end - row_start
            ns_block = col_end - col_start

            nl_c = nl_block // scale_factor
            ns_c = ns_block // scale_factor

            if nl_c < 2 or ns_c < 2:
                continue

            coarse_points = []
            for i_c in range(nl_c):
                for j_c in range(ns_c):
                    r = int(i_c * scale_factor + scale_factor // 2)
                    c = int(j_c * scale_factor + scale_factor // 2)
                    if r < nl_block and c < ns_block:
                        coarse_points.append((r, c))

            coarse_points = np.array(coarse_points)

            grid_r, grid_c = np.meshgrid(
                np.arange(nl_block),
                np.arange(ns_block),
                indexing="ij"
            )

            for b in range(bands):
                coarse_values = np.array([
                    block_data[r, c, b] for r, c in coarse_points
                ])

                rbf = Rbf(
                    coarse_points[:, 0],
                    coarse_points[:, 1],
                    coarse_values,
                    function="multiquadric"
                )

                interp_vals = rbf(
                    grid_r.ravel(),
                    grid_c.ravel()
                ).reshape(nl_block, ns_block)

                result[
                    row_start:row_end,
                    col_start:col_end,
                    b
                ] = interp_vals

            print(f"TPS block ({block_row}, {block_col}) done")

    return result


