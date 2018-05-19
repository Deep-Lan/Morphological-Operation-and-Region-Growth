from skimage import filters, morphology, measure, color
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def OtsuThreshold(img, remove_background=False):
    # histogaram
    bins = np.arange(256)
    hist, _ = np.histogram(img, np.hstack((bins, np.array([256]))))

    # remove background
    black_pixel_num = 0
    if remove_background:
        black_pixel_num = hist[0]
        hist[0] = 0

    # solve otsu threshold
    num_pixel = img.size - black_pixel_num
    hist_norm = hist / num_pixel
    max_delta2 = 0
    threshold = None
    for T in range(255):
        mu0 = 0
        mu1 = 0
        omega0 = np.sum(hist_norm[0:T + 1])
        omega1 = 1 - omega0
        for i in range(T + 1):
            mu0 = mu0 + i * hist_norm[i]
        if omega0 != 0:
            mu0 = mu0 / omega0
        for i in range(T + 1, 256):
            mu1 = mu1 + i * hist_norm[i]
        if omega1 != 0:
            mu1 = mu1 / omega1
        delta2 = omega0 * omega1 * (mu0 - mu1) ** 2
        if max_delta2 < delta2:
            max_delta2 = delta2
            threshold = T
    return threshold


def main():
    # read image
    img = np.array(Image.open('cell2.BMP').convert('L'))

    # filtering
    img_mean = filters.rank.mean(img, morphology.disk(6))
    img_median = filters.median(img, morphology.disk(6))
    img_opening = morphology.opening(img, morphology.disk(6))
    img_closing = morphology.closing(img, morphology.disk(6))

    # otsu threshold segmentation
    threshold = OtsuThreshold(img_closing)
    img_otsu = img_closing.copy()
    img_otsu[img_closing > threshold] = 0
    img_otsu[img_closing <= threshold] = 1

    # remove small object
    img_label = measure.label(img_otsu, 8)
    img_remove = morphology.remove_small_objects(img_label, 2000)
    img_remove[img_remove > 0] = 1

    # mark region
    img_label = measure.label(img_remove, 8)

    # measure the ratio of nucleolus and nucleus
    ratio_area = []
    for i in range(1, 1+img_label.max()):
        img_object = (img_label == i) * img_closing
        threshold = OtsuThreshold(img_object, True)
        bins = np.arange(256)
        hist, _ = np.histogram(img_object, np.hstack((bins, np.array([256]))))
        nucleus = np.sum(hist[1:])
        nucleolus = np.sum(hist[1:threshold])
        ratio = nucleolus/nucleus
        print('ratio:', ratio)
        ratio_area.append(ratio)

    # show images
    plt.figure(1), plt.imshow(img_median, 'gray'), plt.title('median filter')
    plt.figure(2), plt.imshow(img_mean, 'gray'), plt.title('mean filter')
    plt.figure(3), plt.imshow(img_opening, 'gray'), plt.title('opening filter')
    plt.figure(4), plt.imshow(img_closing, 'gray'), plt.title('closing filter')
    plt.figure(5), plt.imshow(img_otsu, 'gray'), plt.title('otsu threshold segmentation')
    plt.figure(6), plt.imshow(img_remove, 'gray'), plt.title('remove small object')
    plt.figure(7), plt.imshow(color.label2rgb(img_label), 'gray'), plt.title('labeled region')
    plt.show()


if __name__ == '__main__':
    main()