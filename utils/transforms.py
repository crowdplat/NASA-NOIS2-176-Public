import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import numbers
from abc import abstractmethod
from typing import Tuple
import random





# Potsdam
ClassesColors = {
    (255, 255, 255): 0, # impervious_surfaces
    (0, 0, 255): 1, # building
    (0, 255, 255): 2, # low_vegetation
    (0, 255, 0): 3, # tree
    (255, 255, 0): 4, # car
    (255, 0, 0): 5 # background
    }

potsdam_map = {v: k for k,v in ClassesColors.items()}

UDD_colors = {
    (0, 255, 0): 0,  # vegetation
    (255, 0, 0): 1,  # building
    (0, 0, 255): 2,  # road
    (128, 128, 0): 3,  # car
    (128, 128, 128): 4,  # other
}

UDD_map = {v: k for k,v in UDD_colors.items()}


def get_class_color(color):
    try:
        return ClassesColors[color]
    except KeyError:
        r, g, b = color
        if r > 200 and g > 200 and b > 200:
            return 0
        elif r < 50 and g < 50 and b > 200:
            return 1
        elif r < 50 and g > 200 and b > 200:
            return 2
        elif r < 50 and g > 200 and b < 50:
            return 3
        elif r > 200 and g > 200 and b < 50:
            return 4
        else:
            return 5

class ToClassLabels(object):
    def __call__(self, segmented_image):
        if torch.is_tensor(segmented_image):
            return segmented_image
        w, h = segmented_image.size
        ret = torch.zeros((h, w), dtype=torch.long)
        for i in range(w):
            for j in range(h):
                color = segmented_image.getpixel((i, j))
                ret[j, i] = get_class_color(color)
                # closest_color = min(list(ClassesColors.keys()), key=lambda x: np.linalg.norm(np.subtract(x, color)))
                # ret[j,i] = ClassesColors[closest_color]
                # ret[j, i] = ClassesColors[color]
        return ret

def get_padding(image):
    image_w, image_h = image.size
    width = 2048
    height= 1024
    w_padding = width - image_w
    h_padding = height - image_h
    l_pad = r_pad = w_padding // 2
    r_pad = width - (image_w + r_pad + l_pad)
    t_pad = b_pad = h_padding // 2
    b_pad = height - (image_h + t_pad + b_pad)
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class SegmentationTensorTransform:
    """Abstract class for transformation on a pytorch Tensor."""
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, image: Tensor, mask: Tensor):
        pass


class SegmentationImageTransform:
    """Abstract class for transformations on a PIL Image."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, image: Image.Image, mask: Image.Image):
        pass


class ToTensor(SegmentationImageTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        """
        :param image: must be in mode 'RGB'
        :param mask: must be in 'L'
        :return: tensors of the image(float) and mask(long)
        """
        if image.mode != 'RGB':
            raise ValueError(f"'image' must be in mode 'RGB'. got '{image.mode}'")
        if mask.mode not in ['L', 'I']:
            raise ValueError(f"'mask' must be in mode 'L' or 'I'. got '{mask.mode}'")

        image = torch.from_numpy(np.array(image))
        image = image.permute(2, 0, 1)
        image = image.float() / 255.

        mask = torch.from_numpy(np.array(mask))
        mask = mask.long()

        return image, mask


class Resize(SegmentationImageTransform):
    """Resize the image"""

    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.resize((self.w, self.h), resample=Image.BILINEAR)
        mask = mask.resize((self.w, self.h), resample=Image.NEAREST)
        return image, mask


class RandomAffine(SegmentationImageTransform):
    def __init__(self, angle_max: int = 0, translate_max: float = 0, scale_max: float = 1,
                 shear_max: int = 0, fillcolor: int = 0):
        super().__init__()
        if angle_max < 0 or angle_max > 180:
            ValueError(f"'angle_max' should be between 0 to 180, got {angle_max}")
        if translate_max < 0 or translate_max > 1:
            ValueError(f"'translate_max' should be between 0 to 1, got {translate_max}")
        if scale_max < 1 or scale_max > 4:
            ValueError(f"'scale_max' should be between 1 to 4, got {scale_max}")
        if shear_max < 0 or shear_max > 180:
            ValueError(f"'shear_max' should be between 0 to 180, got {shear_max}")
        if fillcolor < 0 or fillcolor > 255:
            ValueError(f"'fillcolor should be between 0 to 255, got {fillcolor}")
        self.angle_max = angle_max
        self.translate_max = translate_max
        self.scale_max = scale_max
        self.shear_max = shear_max
        self.fillcolor = fillcolor

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        angle = random.randint(-self.angle_max, self.angle_max)
        translate_w = random.randint(-int(self.translate_max * w), int(self.translate_max * w))
        translate_h = random.randint(-int(self.translate_max * h), int(self.translate_max * h))
        scale = random.uniform(1 / self.scale_max, self.scale_max)
        shear = random.randint(-self.shear_max, self.shear_max)
        image = F.affine(
            image,
            angle=angle,
            translate=[translate_w, translate_h],
            scale=scale,
            shear=shear,
            resample=Image.BILINEAR,
        )
        mask = F.affine(
            mask,
            angle=angle,
            translate=[translate_w, translate_h],
            scale=scale,
            shear=shear,
            resample=Image.NEAREST,
            fillcolor=self.fillcolor,  # This is useful with ignore_index in loss
        )
        return image, mask


class RandomFlip(SegmentationImageTransform):
    def __init__(self, p_hflip: float = 0, p_vflip: float = 0):
        super().__init__()
        if p_hflip < 0 or p_hflip > 1:
            raise ValueError(f"'p_hflip needs to be in [0,1] got {p_hflip}")
        if p_vflip < 0 or p_vflip > 1:
            raise ValueError(f"'p_vflip needs to be in [0,1] got {p_vflip}")
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() <= self.p_hflip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() <= self.p_vflip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class RandomCrop(SegmentationImageTransform):
    def __init__(self, h_crop: int, w_crop: int):
        super().__init__()
        self.h_crop = h_crop
        self.w_crop = w_crop

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        if w < self.w_crop or h < self.h_crop:
            raise ValueError(f"image dimensions {(h, w)} smaller then crop dimensions {(self.h_crop, self.w_crop)}.")
        top = random.randint(0, h - self.h_crop)
        left = random.randint(0, w - self.w_crop)
        image = TF.crop(image, top, left, self.h_crop, self.w_crop)
        mask = TF.crop(mask, top, left, self.h_crop, self.w_crop)
        return image, mask


class BasicTransform(SegmentationImageTransform):
    """Basic for segmentation transform."""

    def __init__(self, h_out: int = None, w_out: int = None, crop: bool = False):
        """
        :param h_out:
        :param w_out:
        :param crop: if the image will be cropped instead of interpulated, will use CenterCrop.
        in this case, the resizing should be strictly downscaling
        """
        super().__init__()
        self.resize = None
        self.crop = None
        if h_out and w_out:
            if crop:
                self.crop = transforms.CenterCrop(size=(h_out, w_out))
            else:
                self.resize = Resize(h_out, w_out)

        self.to_tensor = ToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        """expecting a PIL image and mask."""
        if self.resize is not None:
            image, mask = self.resize(image, mask)
        if self.crop is not None:
            image = self.crop(image)
            mask = self.crop(mask)

        return self.to_tensor(image, mask)


class AugmentationTransform(SegmentationImageTransform):
    def __init__(self,
                 # crop
                 crop: bool = False,
                 # resize
                 h_out: int = None, w_out: int = None,
                 # color jitter
                 color_jitter: bool = False, brightness: float = 0.4, contrast: float = 0.5, saturation: float = 1,
                 hue: float = 0.1,
                 # flip
                 flip: bool = False, p_hflip: float = 0.5, p_vflip: float = 0.5,
                 # affine translation
                 affine: bool = False, angle_max: int = 180, translate_max: float = 0.1, scale_max: float = 1,
                 shear_max: int = 10, fillcolor: int = 255,
                 ):
        super().__init__()
        self.resize = None
        if h_out and w_out:
            if crop:
                self.resize = RandomCrop(h_out, w_out)
            else:
                self.resize = Resize(h_out, w_out)
        self.color_jitter = None
        if color_jitter:
            self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.flip = None
        if flip:
            self.flip = RandomFlip(p_hflip, p_vflip)
        self.affine = None
        if affine:
            self.affine = RandomAffine(angle_max, translate_max, scale_max, shear_max, fillcolor)
        self.to_tensor = ToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.resize:
            image, mask = self.resize(image, mask)
        if self.color_jitter:
            image = self.color_jitter(image)
        if self.flip:
            image, mask = self.flip(image, mask)
        if self.affine:
            image, mask = self.affine(image, mask)
        return self.to_tensor(image, mask)


class NewPad(SegmentationImageTransform):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): img to be padded.

        Returns:
            Padded image.
        """
        if torch.is_tensor(img):
            assert img.shape == (1024, 2048)
            return img
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}'. \
            format(self.fill, self.padding_mode)


def pred_2_img(pred, output_loc='data/output/out.png', show=False):
    label_mask = pred.numpy()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(5):
        r[label_mask == ll] = UDD_map[ll][0]
        g[label_mask == ll] = UDD_map[ll][1]
        b[label_mask == ll] = UDD_map[ll][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if show:
        plt.imshow(rgb)
        plt.show()
    else:
        matplotlib.image.imsave(output_loc, rgb)