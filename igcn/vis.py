import cv2
import numpy as np


class FilterPlot():
    def __init__(self, no_g, f_size, channels, h_space=20, v_space=5, ion=True):
        self.h_space = h_space
        self.v_space = v_space
        self.f_space = 5
        f_space = self.f_space
        self.channels = channels
        self.width = 2 * + h_space + (1 + no_g) * f_size + no_g * self.f_space
        self.height = channels * (f_size + v_space)
        self.no_g = no_g
        self.f_size = f_size
        self.img = np.ones((self.height, self.width))
        self.ion = int(ion)
        self.counter = -1
        self.limit = 6

        self.filt_corner = [(h_space // 2, (f_size + v_space) * i + v_space // 2) for i in range(channels)]
        self.filt = [(slice(y, y + f_size), slice(x, x + f_size)) for x, y in self.filt_corner]

        self.enh_filt_corner = [((f_size + f_space) * j + h_space // 2 + h_space + f_size,
                                 (f_size + v_space) * i + v_space // 2)
                                for j in range(no_g)
                                for i in range(channels)]
        self.enh_filt = [(slice(y, y + f_size), slice(x, x + f_size)) for x, y in self.enh_filt_corner]

    def update(self, filt, enh_filt, params, net_type="igcn"):
        self.counter += 1
        if self.counter % self.limit != 0:
            return

        filt, enh_filt = self._normalise(filt, enh_filt, net_type)
        for slc, sub_img in zip(self.filt, filt):
            # print("Subimg", sub_img.shape, sub_img.min(), sub_img.max(), sub_img.mean())
            self.img[slc] = sub_img

        for slc, sub_img in zip(self.enh_filt, enh_filt):
            self.img[slc] = sub_img

        resized_img = cv2.resize(self.img, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        if params is not None:
            for (ti, li), (x, y) in zip(params.transpose(1, 0), self.enh_filt_corner[::self.channels]):
                cv2.putText(resized_img, "t={:.2f} l={:.2f}.".format(ti, li),
                            (x*10-20, y*10-5), 1, 1, 0, 2)
        # cv2.imshow("Filter", resized_img)
        cv2.imwrite(self._get_next_fname(net_type), resized_img*255)
        cv2.waitKey(self.ion)

    def _normalise(self, filt, enh_filt, net_type):
        if net_type == "gcn":
            filt = filt[:, 0].squeeze()
        max_val = np.max(np.vstack((filt, enh_filt)))
        min_val = np.min(np.vstack((filt, enh_filt)))
        return (filt - min_val) / (max_val - min_val), (enh_filt - min_val) / (max_val - min_val)

    def _get_next_fname(self, net_type):
        count = self.counter // self.limit
        fname = "figs/filters/weightsgabors/" + net_type + "_" + str(count) + ".png"
        return fname
