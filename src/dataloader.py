import itertools

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology


def _ceil_div(num, den):
    num = int(num)
    den = int(den)
    return (num + den - 1) // den


class dataloder:
    def __init__(self, patch_size, stride, target_size, includebg=False):
        # For now consider (h, w, d)
        assert len(patch_size) == 3
        assert len(stride) == 3
        assert len(target_size) == 3
        self.patch_size = patch_size
        self.stride = stride
        self.target_size = target_size
        self.includebg = includebg

    def _slides_to_patches(self, pancreas, lesion=None):
        assert lesion is None or pancreas.shape == lesion.shape
        pan = pancreas
        les = lesion
        patch_size = self.patch_size
        pad_size = (
            (patch_size[0], patch_size[0]),
            (patch_size[1], patch_size[1]),
            (patch_size[2], patch_size[2]),
        )

        stride = self.stride

        pan = np.pad(pan, pad_size, "constant")
        total = (
            len(range(0, pan.shape[0], stride[0]))
            * len(range(0, pan.shape[1], stride[1]))
            * len(range(0, pan.shape[2], stride[2]))
        )
        X = np.zeros((total,) + patch_size)
        y = None
        if les is not None:
            les = np.pad(les, pad_size, "constant")
            y = np.zeros(total)
        for (sx, sy, sz), index in zip(
            itertools.product(
                range(0, pan.shape[0], stride[0]),
                range(0, pan.shape[1], stride[1]),
                range(0, pan.shape[2], stride[2]),
            ),
            range(total),
        ):
            ex = np.min([sx + patch_size[0], pan.shape[0]])
            ey = np.min([sy + patch_size[1], pan.shape[1]])
            ez = np.min([sz + patch_size[2], pan.shape[2]])
            X[index][: ex - sx, : ey - sy, : ez - sz] = pan[sx:ex, sy:ey, sz:ez]
            if lesion is not None:
                target_size = self.target_size
                tx = (patch_size[0] - target_size[0]) // 2
                ty = (patch_size[1] - target_size[1]) // 2
                tz = (patch_size[2] - target_size[2]) // 2
                img_les = np.zeros(patch_size)
                img_les[: ex - sx, : ey - sy, : ez - sz] = les[sx:ex, sy:ey, sz:ez]
                y[index] = (
                    np.sum(
                        img_les[
                            tx : tx + target_size[0],
                            ty : ty + target_size[1],
                            tz : tz + target_size[2],
                        ]
                    )
                    > 0
                ).astype(int)

        return X, y

    def _split_to_patches(self, pancreas, lesion=None):
        if lesion is None:
            X = []
            for pan in pancreas:
                x, _ = self._slides_to_patches(pan, None)
                X.append(x)
            X = np.concatenate(X)
            return X, None
        else:
            X = []
            Y = []
            for pan, les in zip(pancreas, lesion):
                x, y = self._slides_to_patches(pan, les)
                X.append(x)
                Y.append(y)
            X = np.concatenate(X)
            Y = np.concatenate(Y)
            return X, Y

    def construct(self, pancreas, lesion=None):
        if type(pancreas) is list:
            X, Y = self._split_to_patches(pancreas, lesion)
        else:
            X, Y = self._slides_to_patches(pancreas, lesion)

        return X, Y

    def get_score(self, img_shape, results):
        target_size = self.target_size
        patch_size = self.patch_size
        sp = (
            img_shape[0] + 2 * patch_size[0],
            img_shape[1] + 2 * patch_size[1],
            img_shape[2] + 2 * patch_size[2],
        )
        stride = self.stride
        num = (
            _ceil_div(sp[0], stride[0]),
            _ceil_div(sp[1], stride[1]),
            _ceil_div(sp[2], stride[2]),
        )

        score = np.zeros(sp)
        factor = np.zeros(sp)
        index = 0

        tx = (patch_size[0] - target_size[0]) // 2
        ty = (patch_size[1] - target_size[1]) // 2
        tz = (patch_size[2] - target_size[2]) // 2
        assert num[0] * num[1] * num[2] == results.shape[0]
        for (sx, sy, sz) in itertools.product(
            range(0, sp[0], stride[0]),
            range(0, sp[1], stride[1]),
            range(0, sp[2], stride[2]),
        ):
            ex = int(np.min([sx + tx + target_size[0], sp[0]]))
            ey = int(np.min([sy + ty + target_size[1], sp[1]]))
            ez = int(np.min([sz + tz + target_size[2], sp[2]]))
            score[sx + tx : ex, sy + ty : ey, sz + tz : ez] = (
                score[sx + tx : ex, sy + ty : ey, sz + tz : ez] + results[index]
            )
            factor[sx + tx : ex, sy + ty : ey, sz + tz : ez] = (
                factor[sx + tx : ex, sy + ty : ey, sz + tz : ez] + 1
            )
            index = index + 1
        print(index)
        score[factor != 0] = score[factor != 0] / factor[factor != 0]
        return score[
            patch_size[0] : patch_size[0] + img_shape[0],
            patch_size[1] : patch_size[1] + img_shape[1],
            patch_size[2] : patch_size[2] + img_shape[2],
        ]
