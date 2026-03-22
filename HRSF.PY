import os
import numpy as np
from HRSF_func import *
from sklearn.cluster import KMeans
from datetime import datetime
from skimage.transform import downscale_local_mean
import os
from datetime import datetime
from sklearn.cluster import KMeans
from skimage.transform import downscale_local_mean, resize


# ----------------------------
# 固定参数设置
# ----------------------------
scale_factor = 17
class_num = 5
win_size = 15
object_RC_percent = 5
similar_win_size = 31
similar_num = 20
min_val = 0.0
max_val = 10000.0

# 固定路径（不参与遍历）
F_tb_path = r"f:\CODE\flood\1-s22CLIP\s22019-03-06_sim_crop.tif"
F_tb_objects_path = r"f:\CODE\flood\2-sam\19-03-06sam.tif"
F_tb_class_path = r"f:\CODE\flood\2-kmeans\19-03-06-KMeans-5.tif"
C_tb_path = r"f:\CODE\flood\1-modisCLIP\03-06pre\MODIS2019-03-06_crop_sim.tif"

# 需要遍历的目录（即 C_tp_path 所在的目录）
C_tp_dir = r"f:\CODE\flood\1-modisCLIP\03-06pre\1"

# 输出保存目录
output_dir = r"F:\CODE\flood\3-result"
os.makedirs(output_dir, exist_ok=True)


# ----------------------------
# 封装主函数
# ----------------------------
def process_one_tif(C_tp_path):
    """对单个 C_tp 文件执行 OBSUM 下采样融合"""
    print(f"\nProcessing: {C_tp_path}")

    # 输出路径
    file_name = os.path.basename(C_tp_path)
    F_tp_OBSUM_path = os.path.join(output_dir, file_name)

    # 1. 加载输入影像
    F_tb, F_tb_profile = read_raster(F_tb_path)
    C_tb = read_raster(C_tb_path)[0]
    C_tb_coarse = downscale_local_mean(C_tb, factors=(scale_factor, scale_factor, 1))
    C_tp = read_raster(C_tp_path)[0]
    C_tp_coarse = downscale_local_mean(C_tp, factors=(scale_factor, scale_factor, 1))
    F_tb_objects = read_raster(F_tb_objects_path)[0][:, :, 0]

    # 2. 加载或生成分类图
    if not os.path.exists(F_tb_class_path):
        kmeans = KMeans(n_clusters=class_num, max_iter=300)
        kmeans.fit(F_tb.reshape(-1, F_tb.shape[2]))
        F_tb_class = kmeans.labels_.reshape(F_tb.shape[0], F_tb.shape[1])
        write_raster(np.expand_dims(F_tb_class, axis=2), F_tb_profile, F_tb_class_path)
        print("Classified the fine image using K-Means algorithm!")
    else:
        F_tb_class = read_raster(F_tb_class_path)[0][:, :, 0]
        print("Loaded the land-cover classification map!")

    # 3. 执行 OBSUM 融合
    time0 = datetime.now()
    obsum = HRSF(
        F_tb, C_tb_coarse, C_tp_coarse, F_tb_class, F_tb_objects,
        class_num=class_num, scale_factor=scale_factor,
        win_size=win_size, OL_RC_percent=object_RC_percent,
        similar_win_size=similar_win_size, similar_num=similar_num,
        min_val=min_val, max_val=max_val
    )
    F_tp_OBSUM = obsum.object_based_spatial_unmixing()
    time1 = datetime.now()
    print(f"Used {(time1 - time0).total_seconds():.2f} seconds!")

    # 4. 保存结果
    write_raster(F_tp_OBSUM, F_tb_profile, F_tp_OBSUM_path)
    print(f"Saved: {F_tp_OBSUM_path}")


# ----------------------------
# 遍历目录，批量处理所有 C_tp 文件
# ----------------------------
def batch_process_directory(root_dir):
    """递归遍历目录，处理所有 .tif 文件"""
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".tif"):
                C_tp_path = os.path.join(root, file)
                process_one_tif(C_tp_path)


if __name__ == "__main__":
    batch_process_directory(C_tp_dir)