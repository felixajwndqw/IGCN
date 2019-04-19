import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np



class FilterPlot():
    def __init__(self, no_g):
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.filt = self.fig.add_subplot(no_g, 2, 1)
        self.enh_filt = [self.fig.add_subplot(no_g, 2, 2 + i * 2) for i in range(no_g)]
        self.setup_ax([self.filt, *self.enh_filt])
        self.fig.show()

    def update(self, filt, enh_filt, params):
        self.filt.imshow(filt)
        for e_ax, e_img, (ti, li) in zip(self.enh_filt, enh_filt, params.transpose(1, 0)):
            # print(e_img.shape)
            e_ax.imshow(e_img)
            e_ax.set_title(r"$\theta = {:.2f}. \lambda = {:.2f}.$".format(ti, li))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setup_ax(self, ax):
        if type(ax) is not list:
            ax = [ax]
        
        for axi in ax:
            axi.set_yticks([])
            axi.set_xticks([])


class FilterPlotNew():
    def __init__(self, no_g, f_size, h_space=20, v_space=5):
        self.h_space = h_space
        self.v_space = v_space
        self.width = 2 * (f_size + h_space)
        self.height = no_g * (f_size + v_space)
        self.no_g = no_g
        self.f_size = f_size
        self.img = np.ones((self.height, self.width))
        self.filt = (slice((self.height - f_size) // 2, (self.height - f_size) // 2 + f_size),
                     slice(h_space // 2, h_space // 2 + f_size))
        self.enh_filt_corner = [((self.width + h_space) // 2, (f_size + v_space) * i + v_space // 2) for i in range(no_g)]
        self.enh_filt = [(slice(y, y + f_size), slice(x, x + f_size)) for x, y in self.enh_filt_corner]
    
    def update(self, filt, enh_filt, params):
        self.img[self.filt] = ((filt + 1) / 2)
        for slc, sub_img in zip(self.enh_filt, enh_filt):
            self.img[slc] = (sub_img + 1) / 2

        resized_img = cv2.resize(self.img, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        for (ti, li), (x, y) in zip(params.transpose(1, 0), self.enh_filt_corner):
            cv2.putText(resized_img, "t = {:.2f}. l = {:.2f}.".format(ti, li),
                        (x*10-20, y*10-5), 1, 1, 0, 2)
        cv2.imshow("Filter", resized_img)
        cv2.waitKey(1)





# class FilterImagePlot():
#     def __init__(self, no_g):
#         self.fig = plt.figure()
#         self.img = self.fig.add_subplot(no_g, 5, 1)
#         self.filt = self.fig.add_subplot(no_g, 5, 2)
#         self.gabor = [self.fig.add_subplot(no_g, 5, 3 + i * 5) for i in range(no_g)]
#         self.enh_filt = [self.fig.add_subplot(no_g, 5, 4 + i * 5) for i in range(no_g)]
#         self.out = [self.fig.add_subplot(no_g, 5, 5 + i * 5) for i in range(no_g)]

#     def update(self, img, filt, gabor, enh_filt, out, params):
#         self.img.imshow(img)
#         self.filt.imshow(filt)
#         for g_ax, e_ax, o_ax, (ti, li) in zip(self.gabor, self.enh_filt, self.out, params.transpose(0, 1)):
#             g_ax.imshow(gabor)
#             g_ax.set_title(r"$\theta = {.2f}. \lambda = {.2f}.$".format(ti, li))
#             e_ax.imshow(enh_filt)
#             o_ax.imshow(out)
#         self.fig.canvas.draw()
