from __future__ import print_function, division
import pandas as pd
from sphinx.directives import patches

from meye import MEImage
from matplotlib import pyplot
from pylab import *
from scipy import signal as sg
from scipy.ndimage.filters import maximum_filter

plt = pyplot


def compute_derived(signpost_image, variable):
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    if variable == "y":
        return sg.convolve2d(signpost_image, sobel_kernel.transpose(), "same")
    elif variable == "x":
        return sg.convolve2d(signpost_image, sobel_kernel, "same")


def compute_products_of_derivatives_at_every_pixel(dx, dy):
    return dx ** 2, dy ** 2, dx * dy


def compute_sums_of_products_of_derivatives(num_frame, i_x2, i_y2, i_xy):
    k = num_frame
    frame = np.ones((k, k)) / k * k
    s_x2 = sg.convolve2d(i_x2, frame, "same")
    s_y2 = sg.convolve2d(i_y2, frame, "same")
    s_xy = sg.convolve2d(i_xy, frame, "same")
    return s_x2, s_y2, s_xy


def do_algorthim(signpost_image):
    # derivative by x
    pdx = compute_derived(signpost_image, "x")
    # derivative by y
    pdy = compute_derived(signpost_image, "y")
    # compute products of derivatives at every pixel
    i_x2, i_y2, i_xy = compute_products_of_derivatives_at_every_pixel(pdx, pdy)
    # compute the sums of the products of derivatives at each pixel
    k = 3
    s_x2, s_y2, s_xy = compute_sums_of_products_of_derivatives(k, i_x2, i_y2, i_xy)

    final_r_matrix = (np.multiply(s_x2, s_y2) - np.multiply(s_xy, s_xy)) - 0.04 * ((s_x2 + s_y2) ** 2)

    my_image = maximum_filter(final_r_matrix, 20) - final_r_matrix
    return my_image


def draw(path_photo, rect):
    plt.rcParams['image.cmap'] = 'gray'
    originl_image = MEImage.from_file(path_photo)
    signpost_image = originl_image.im[rect[2]:rect[3], rect[0]:rect[1]]
    my_image = do_algorthim(signpost_image)

    w, h = signpost_image.shape
    apslon = 10
    for y, px in enumerate(my_image):
        if (y > apslon and y < h - apslon):
            for x, py in enumerate(px):
                if py == 0 and (x > apslon and x < w - apslon):
                    signpost_image[y][x] = 255
    return signpost_image


df = pd.read_pickle("store.pickle")

rect_49 = df.iloc[2].prevRect
img_49 = df.iloc[2].prevImage
prev_img = draw(img_49, rect_49)

rect_50 = df.iloc[2].currRect
img_50 = df.iloc[2].currImage
curr_img = draw(img_50, rect_50)

# plt.imshow(prev_img, cmap='gray', origin='lower')
# plt.show()
# colorbar()
ego_motion = df['egoMotion'][2]
np.delete(ego_motion, 3,1)
print(ego_motion)

plt.imshow(curr_img, cmap='gray', origin='lower')
plt.show()
