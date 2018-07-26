from utils import check_folder
import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm

def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='hw', help='dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')

    return parser.parse_args()

def make_edge_smooth(dataset_name, img_size) :
    check_folder('./dataset/{}/{}'.format(dataset_name, 'trainB_smooth'))

    file_list = glob('./dataset/{}/{}/*.*'.format(dataset_name, 'trainB'))
    save_dir = './dataset/{}/trainB_smooth'.format(dataset_name)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        h, w = edges.shape

        gauss_img = np.copy(bgr_img)
        for i in range(kernel_size // 2, h - kernel_size // 2):
            for j in range(kernel_size // 2, w - kernel_size // 2):
                if dilation[i, j] != 0:  # gaussian blur to only edge
                    gauss_img[i, j, 0] = np.sum(np.multiply(bgr_img[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1, 0], gauss))
                    gauss_img[i, j, 1] = np.sum(np.multiply(bgr_img[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1, 1], gauss))
                    gauss_img[i, j, 2] = np.sum(np.multiply(bgr_img[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1, 2], gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth(args.dataset, args.img_size)


if __name__ == '__main__':
    main()