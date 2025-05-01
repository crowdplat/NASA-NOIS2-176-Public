from PIL import Image
import os
import glob
from pathlib import Path
import math

# change extensions
def split(images_dir, labels_dir, output_dir):
    for dir in [images_dir, labels_dir]:
        images = glob.glob(os.path.join(dir, '*.JPG')) + glob.glob(os.path.join(dir, '*.png'))
        for image in images:
            im = Image.open(image)
            imgwidth, imgheight = im.size
            image_name = image.split('/')[-1].strip('.JPG')
            new_image_dir = os.path.join(output_dir, dir.split('/')[-1])
            Path(new_image_dir).mkdir(parents=True, exist_ok=True)
            boxes = create_boxes(2048, 1024, imgwidth, imgheight)
            for i, box in enumerate(boxes):
                cropped = im.crop(box)
                cropped.save(os.path.join(new_image_dir, image_name + '_box' + str(i)) + '.png')


def potsdam_split():
    #(left, upper, right, lower) -
    box1 = (0,    0,    2048, 1024)
    box2 = (2000, 1000, 4048, 2024)
    box3 = (3952, 2000, 6000, 3024)
    box4 = (0,    3000, 2048, 4024)
    box5 = (2000, 4000, 4048, 5024)
    box6 = (3953, 4976, 6000, 6000)
    boxes = [box1, box2, box3, box4, box5, box6]
    images_dir = '2_Ortho_RGB'
    labels_dir = 'labels'
    output_dir = 'output'


def create_boxes(width, height, imgwidth, imgheight):
    """
    boxes to split image
    :param width: width of cropped box
    :param height: height of cropped box
    :param imgwidth: image width
    :param imgheight: image height
    """
    # im = Image.open(input)
    # imgwidth, imgheight = im.size

    num_boxes_w = math.ceil(imgwidth/width)
    num_boxes_h = math.ceil(imgheight/height)

    overlap_w = (num_boxes_w * width - imgwidth) // (num_boxes_w - 1)
    overlap_h = (num_boxes_h * height - imgheight) // (num_boxes_h - 1)

    boxes = []
    upper = 0
    for i in range(num_boxes_h):
        upper = max(0, upper - overlap_h)
        lower = min(imgheight, upper + height)
        left = 0

        for j in range(num_boxes_w):
            left = max(0, left - overlap_w)
            right = min(imgwidth, left + width)
            box = (left, upper, right, lower)
            boxes.append(box)
            left = right
        upper = lower

    return boxes


if __name__ == "__main__":
    images_dir = '/home/eladamar/fast_scnn/data/UDD5/train/src'
    labels_dir = '/home/eladamar/fast_scnn/data/UDD5/train/gt'
    output_dir = '/home/eladamar/fast_scnn/data/UDD5/train/splitted'
    split(images_dir, labels_dir, output_dir)
