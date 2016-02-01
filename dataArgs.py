# coding:utf-8
import cv2
import numpy as np
import math
import copy
import random
import warnings


class DataArgumentation(object):

    def __init__(self, srcimg):
        super(DataArgumentation, self).__init__(

        )
        self.src = cv2.bitwise_not(srcimg)  # opencv mat 1ch 背景黒にネガポジ反転
        # 平行移動
        self.transition_levels = [0, 3, 5, 6, 8, 10, 12]
        # 回転角
        self.rotation_theta = [0, 0.1, 0.2, 0.3, 0.4,
                               0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

    def argumentation(self, whiches):
        dst = self.src
        if 2 in whiches:
            # rotation parameter setting
            rotation_vector = [random.randint(
                -1, 1), random.randint(-1, 1), random.randint(-1, 1)]
            roi = [[5, 5], [60, 5], [60, 60], [5, 60]]
            dst = self.rotation(dst, rotation_vector, self.rotation_theta[
                                random.randint(0, 7)], roi)
        if 3 in whiches:
            dst = self.transition(dst, self.transition_levels[
                                  random.randint(0, 4)])
        return dst

    # theta = 0.7, 1.0, 1.3
    def rotation(self, src, vector, theta, roi):
        np.seterr(all='warn')
        warnings.filterwarnings('error')
        try:
            rotation_vector = (np.array(vector) /
                               np.linalg.norm(np.array(vector))).tolist()
        except RuntimeWarning:
            return src
        warnings.filterwarnings("default")

        R = np.array([
            [0, -rotation_vector[2], rotation_vector[1]], [rotation_vector[2], 0, -rotation_vector[0]], [-rotation_vector[1], rotation_vector[0], 0]])
        I = np.matrix(np.identity(3))
        M = I + (math.sin(theta) * R) + ((1 - math.cos(theta)) * np.dot(R, R))
        M = np.vstack((M, np.array([0, 0, 0])))
        M = np.c_[M, np.array([0, 0, 0, 1])]
        # listの代入は参照渡しなのでオリジナルのroiをdeepcopyしておく
        roi_original = copy.deepcopy(roi)
        # 4次元座標にしておく
        lt, rx, rb, lb = roi[0], roi[1], roi[2], roi[3]
        lt.extend([0, 0])
        rx.extend([0, 0])
        rb.extend([0, 0])
        lb.extend([0, 0])

        lt, rx, rb, lb = np.array([lt]).T, np.array(
            [rx]).T, np.array([rb]).T, np.array([lb]).T
        lt_r = np.dot(M, lt)
        rx_r = np.dot(M, rx)
        rb_r = np.dot(M, rb)
        lb_r = np.dot(M, lb)

        rotation_roi = [[lt_r[0, 0], lt_r[1, 0]], [rx_r[0, 0], rx_r[1, 0]], [
            rb_r[0, 0], rb_r[1, 0]], [lb_r[0, 0], lb_r[1, 0]]]

        rotation_roi = np.array(rotation_roi)
        # rotation_roiの重心を原点に
        rotation_roi_center = (rotation_roi[
                               0] / 4) + (rotation_roi[1] / 4) + (rotation_roi[2] / 4) + (rotation_roi[3] / 4)
        img_center = [src.shape[1] / 2, src.shape[0] / 2]
        diff = rotation_roi_center - img_center
        rotation_roi = [rotation_roi[0] - diff, rotation_roi[1] -
                        diff, rotation_roi[2] - diff, rotation_roi[3] - diff]

        perspective1 = np.float32(roi_original)
        perspective2 = np.float32(rotation_roi)

        psp_matrix = cv2.getPerspectiveTransform(
            perspective1, perspective2)
        size = tuple(np.array([src.shape[1], src.shape[0]]))
        dst = cv2.warpPerspective(src, psp_matrix, size)

        return dst

    def transition(self, src, level):
        size = tuple(np.array([src.shape[1], src.shape[0]]))
        if random.randint(0, 1) == 0:
            move_x = level
        else:
            move_x = level * -1
        if random.randint(0, 1) == 0:
            move_y = level
        else:
            move_y = level * -1
        matrix = [
            [1,   0, move_x],
            [0,   1, move_y]
        ]
        affine_matrix = np.float32(matrix)
        img_afn = cv2.warpAffine(src, affine_matrix,
                                 size, flags=cv2.INTER_LINEAR)
        return img_afn
