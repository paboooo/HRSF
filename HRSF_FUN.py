import numpy as np
from scipy.optimize import lsq_linear
from skimage.transform import downscale_local_mean, resize


class HRSF:
    def __init__(self, F_tb, C_tb, C_tp, F_tb_class, F_tb_objects, #F_C_tp,
                 class_num=5, scale_factor=17, win_size=15, OL_RC_percent=5, similar_win_size=31, similar_num=20, min_val=0, max_val=10000):
        self.F_tb = F_tb.astype(np.float32)
        self.C_tb = C_tb.astype(np.float32)
        self.C_tp = C_tp.astype(np.float32)
        self.delta_C = self.C_tp - self.C_tb
        self.F_tb_class = F_tb_class
        self.F_tb_objects = F_tb_objects
        #self.F_C_tp = F_C_tp
        self.class_num = class_num
        self.scale_factor = scale_factor
        self.win_size = win_size
        self.OL_RC_percent = OL_RC_percent
        self.similar_win_size = similar_win_size
        self.similar_num = similar_num
        self.min_val = min_val
        self.max_val = max_val

    def refine_classification_using_objects(self):

        refined_class = np.empty(shape=self.F_tb_class.shape, dtype=np.uint8)

        object_indices = np.unique(self.F_tb_objects)
        for object_idx in object_indices:
            object_mask = self.F_tb_objects == object_idx
            object_classes = self.F_tb_class[object_mask]
            if np.count_nonzero(object_mask) == 1:
                object_class = object_classes
            else:
                object_class = np.argmax(np.bincount(object_classes.squeeze()))
            refined_class[object_mask] = object_class

        self.F_tb_class = refined_class
        print(f"Refined land-cover classification map!")

    def calculate_class_fractions(self):

        C_fractions = np.zeros(shape=(self.C_tp.shape[0], self.C_tp.shape[1], self.class_num), dtype=np.float32)
        for row_idx in range(self.C_tp.shape[0]):
            for col_idx in range(self.C_tp.shape[1]):
                F_class_pixels = self.F_tb_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                                 col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor]
                for class_idx in range(self.class_num):
                    pixel_num = np.count_nonzero(F_class_pixels == class_idx)
                    C_fractions[row_idx, col_idx, class_idx] = pixel_num / (self.scale_factor * self.scale_factor)

        return C_fractions

    def unmix_window(self, C_values, C_fractions, lower_bound, upper_bound):

        lsq = lsq_linear(C_fractions, C_values,
                         bounds=(lower_bound, upper_bound), method="bvls", max_iter=100)

        result = lsq.x

        return result

    def calculate_distances_in_coarse_pixel(self):

        rows = np.linspace(start=0, stop=self.scale_factor - 1, num=self.scale_factor)
        cols = np.linspace(start=0, stop=self.scale_factor - 1, num=self.scale_factor)
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        central_row = self.scale_factor // 2
        central_col = self.scale_factor // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        distances = np.concatenate([distances for i in range(self.C_tp.shape[0])], axis=0)
        distances = np.concatenate([distances for i in range(self.C_tp.shape[1])], axis=1)

        # normalize to [1, 1+sqrt(2)]
        distances = 1 + distances / (self.scale_factor // 2)

        return distances

    def calculate_object_homogeneity_index(self):

        object_homo_index = np.zeros(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=np.float32)
        F_tb_objects_pad = np.pad(self.F_tb_objects, pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                                                (self.similar_win_size // 2, self.similar_win_size // 2)),
                                  mode="reflect")
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):
                current_object = self.F_tb_objects[row_idx, col_idx]
                pixel_objects = F_tb_objects_pad[row_idx:row_idx + self.similar_win_size,
                                col_idx:col_idx + self.similar_win_size]
                object_homo_index[row_idx, col_idx] = np.count_nonzero(pixel_objects == current_object) / \
                                                      np.square(self.similar_win_size)

        return object_homo_index


    def calculate_vaild_homogeneity_index(self):
        vaild_homo_index = np.zeros(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=np.float32)

        self.delta_F = resize(self.delta_C, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order=3)
        delta_F_pad = np.pad(self.delta_F,
                            pad_width=((self.scale_factor // 2, self.scale_factor // 2),
                                        (self.scale_factor // 2, self.scale_factor // 2),
                                        (0, 0)),
                            mode="reflect")
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):
                central_pixel_vailds = self.delta_F[row_idx, col_idx, :]
                neighbor_pixel_vailds = delta_F_pad[row_idx:row_idx + self.scale_factor,
                col_idx:col_idx + self.scale_factor, :]
                V = np.mean(np.abs(neighbor_pixel_vailds - central_pixel_vailds), axis=2).flatten()
                vaild_homo_index[row_idx, col_idx] = np.mean(1 / (1 + V))

        return vaild_homo_index

    def calculate_similar_pixel_distances(self):

        rows = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        cols = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        central_row = self.similar_win_size // 2
        central_col = self.similar_win_size // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        # normalize to [1, 1+sqrt(2)]
        distances = 1 + distances / (self.similar_win_size // 2)

        return distances


    def select_similar_pixels(self):

        self.F_C_tp = resize(self.C_tp, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order=3)
        self.delta_F = self.F_C_tp - self.F_tb

        F_tb_pad = np.pad(self.F_tb,
                            pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                     (self.similar_win_size // 2, self.similar_win_size // 2),
                                     (0, 0)),
                            mode="reflect")
        delta_F_pad = np.pad(self.delta_F,
                            pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                        (self.similar_win_size // 2, self.similar_win_size // 2),
                                        (0, 0)),
                            mode="reflect")
        F_tb_similar_weights = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.similar_num), dtype=np.float32)
        F_tb_similar_indices = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.similar_num), dtype=np.uint32)

        distances = self.calculate_similar_pixel_distances().flatten()
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):

                central_pixel_vailds = self.delta_F[row_idx, col_idx, :]
                neighbor_pixel_vailds = delta_F_pad[
                    row_idx:row_idx + self.similar_win_size,
                    col_idx:col_idx + self.similar_win_size,
                    :
                ]
                V = np.mean( np.abs(neighbor_pixel_vailds - central_pixel_vailds), axis=2 ).flatten()
                central_pixel_vals = self.F_tb[ row_idx, col_idx, :]
                neighbor_pixel_vals = F_tb_pad[ row_idx:row_idx + self.similar_win_size, col_idx:col_idx + self.similar_win_size, :]
                D = np.mean( np.abs(neighbor_pixel_vals - central_pixel_vals), axis=2 ).flatten()
                center_idx = (self.similar_win_size ** 2) // 2
                V[center_idx] = np.inf
                D[center_idx] = np.inf
                N = V.size
                rank_V = np.argsort(np.argsort(V)) 
                rank_D = np.argsort(np.argsort(D))

                rank_diff_thresh = self.similar_num
                valid_mask = np.abs(rank_V - rank_D) <= rank_diff_thresh
                score = ((N - rank_V) + (N - rank_D)).astype(np.float32)
                score[~valid_mask] = -np.inf
                similar_indices = np.argsort(-score)[:self.similar_num]
                F_tb_similar_indices[row_idx, col_idx, :] = similar_indices

                similar_distances = 1 + distances[similar_indices] / (self.similar_win_size // 2)
                similar_weights = (1 / similar_distances) / np.sum(1 / similar_distances)

                F_tb_similar_weights[row_idx, col_idx, :] = similar_weights

        return F_tb_similar_indices, F_tb_similar_weights

    def hierarchical_residual_distribution_framework(self):

        F_tp = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.C_tp.shape[2]),
                              dtype=self.C_tp.dtype)

        self.refine_classification_using_objects()
        C_fractions = self.calculate_class_fractions()

        distances_in_C = self.calculate_distances_in_coarse_pixel()
        vaild_homogeneity_index = self.calculate_vaild_homogeneity_index()
        vaild_homogeneity_index = (1-vaild_homogeneity_index) / distances_in_C

        delta_C_pad = np.pad(self.delta_C, pad_width=((self.win_size // 2, self.win_size // 2),
                                                      (self.win_size // 2, self.win_size // 2), (0, 0)), mode="reflect")
        C_fractions_pad = np.pad(C_fractions, pad_width=((self.win_size // 2, self.win_size // 2),
                                                         (self.win_size // 2, self.win_size // 2), (0, 0)),
                                 mode="reflect")
        object_indices = np.unique(self.F_tb_objects)

        F_tb_similar_indices, F_tb_similar_weights = self.select_similar_pixels()


# initial prediction
        for band_idx in range(self.C_tp.shape[2]):
            lower_bound = np.min(delta_C_pad[:, :, band_idx])
            upper_bound = np.max(delta_C_pad[:, :, band_idx])
            in_prediction = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=self.C_tp.dtype)
            for row_idx in range(self.C_tp.shape[0]):
                for col_idx in range(self.C_tp.shape[1]):
                    C_pixels_win = delta_C_pad[row_idx:row_idx + self.win_size, col_idx:col_idx + self.win_size,
                                   band_idx]
                    C_fractions_win = C_fractions_pad[row_idx:row_idx + self.win_size,
                                      col_idx:col_idx + self.win_size, :]

                    F_values = self.unmix_window(C_pixels_win.flatten(),
                                                 C_fractions_win.reshape(C_pixels_win.shape[0] * C_pixels_win.shape[1],
                                                                         self.class_num),
                                                 lower_bound, upper_bound)

                    C_classes = self.F_tb_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                                col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor]

                    for class_idx in range(self.class_num):
                        class_mask = C_classes == class_idx
                        in_prediction[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                        col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor][class_mask] = \
                            F_values[class_idx]

            for object_idx in object_indices:
                object_mask = self.F_tb_objects == object_idx
                object_value = np.mean(in_prediction[object_mask])
                F_tp[:, :, band_idx][object_mask] = self.F_tb[:, :, band_idx][object_mask] + object_value

            F_tp[F_tp > self.max_val] = self.max_val
            F_tp[F_tp < self.min_val] = self.min_val

            C_tp_prediction = downscale_local_mean(F_tp[:, :, band_idx],
                                                   factors=(self.scale_factor, self.scale_factor))
            C_residuals = self.C_tp[:, :, band_idx] - C_tp_prediction
            F_residuals = resize(C_residuals, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order=3)

# OR prediction
            for object_idx in object_indices:
                object_mask = self.F_tb_objects == object_idx
                object_residuals = F_residuals[object_mask]

                residual_indices = vaild_homogeneity_index[object_mask]
                indices = (residual_indices >= np.percentile(residual_indices, 100 - self.OL_RC_percent)).nonzero()[0]
                selected_residuals = object_residuals[indices]

                selected_weights = residual_indices[indices]
                selected_weights = selected_weights / np.sum(selected_weights)
                residual = np.sum(selected_weights * selected_residuals)

                F_tp[:, :, band_idx][object_mask] += residual
# CR prediction

            F_tp[F_tp > self.max_val] = self.max_val
            F_tp[F_tp < self.min_val] = self.min_val

            C_tp_prediction = downscale_local_mean(F_tp[:, :, band_idx],
                                                   factors=(self.scale_factor, self.scale_factor))
            C_residuals = self.C_tp[:, :, band_idx] - C_tp_prediction
            F_residuals = resize(C_residuals, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order=3)

            F_residuals_pad = np.pad(F_residuals,
                                     pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                                (self.similar_win_size // 2, self.similar_win_size // 2)),
                                     mode="reflect")
            for row_idx in range(F_residuals.shape[0]):
                for col_idx in range(F_residuals.shape[1]):
                    neighbor_pixel_residuals = F_residuals_pad[row_idx:row_idx + self.similar_win_size,
                                               col_idx:col_idx + self.similar_win_size]

                    similar_indices = F_tb_similar_indices[row_idx, col_idx, :]
                    similar_residuals = neighbor_pixel_residuals.flatten()[similar_indices]
                    similar_weights = F_tb_similar_weights[row_idx, col_idx, :]

                    residual = np.sum(similar_residuals * similar_weights)

                    F_tp[row_idx, col_idx, band_idx] += residual
            print(f"Finished final prediction of band {band_idx}!")
            F_tp[F_tp > self.max_val] = self.max_val
            F_tp[F_tp < self.min_val] = self.min_val

        return F_tp


