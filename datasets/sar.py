# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import numpy as np
from PIL import Image

from .dataset import Dataset
from .pair_dataset import PairDataset, StillPairDataset

import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import torchvision.transforms.functional as transform

import torch
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from skimage.transform import AffineTransform
import rasterio as rs
import matplotlib.pyplot as plt
import torchvision
from skimage.transform import warp
import numpy as np

def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


def get_image_size(img: Tensor) -> List[int]:

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(get_image_size)
    if isinstance(img, torch.Tensor):
        return F_t.get_image_size(img)

    return F_pil.get_image_size(img)


class RandomAffine(torch.nn.Module):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        fillcolor (sequence or number, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``fill`` instead.
        resample (int, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``interpolation``
                instead.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
            self,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            patch_translate=None,
            interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            fill=0,
            fillcolor=None,
            resample=None,
            center=None,
    ):
        super().__init__()
        torchvision.utils._log_api_usage_once(self)
        if resample is not None:
            warnings.warn(
                "The parameter 'resample' is deprecated since 0.12 and will be removed in 0.14. "
                "Please use 'interpolation' instead."
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        if fillcolor is not None:
            warnings.warn(
                "The parameter 'fillcolor' is deprecated since 0.12 and will be removed in 0.14. "
                "Please use 'fill' instead."
            )
            fill = fillcolor

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if patch_translate is not None:
            _check_sequence_input(patch_translate, "translate", req_sizes=(2,))
            for k in patch_translate:
                for t in k:
                    if not (0.0 <= t <= 1.0):
                        raise ValueError("translation values should be between 0 and 1")
        self.patch_translate = patch_translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int],
            patch_translate: Optional[List[float]],
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)
        if patch_translate is not None:
            min_dx = float(patch_translate[0][0] * img_size[0])
            min_dy = float(patch_translate[1][0] * img_size[1])
            max_dx = float(patch_translate[0][1] * img_size[0])
            max_dy = float(patch_translate[1][1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(min_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(min_dy, max_dy).item()))
            patch_translations = (tx, ty)
        else:
            patch_translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear, patch_translations

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(img)

        angle, translations, scale, shear, patch_location = self.get_params(self.degrees, self.translate, self.scale,
                                                                            self.shear, img_size, self.patch_translate)
        matrix = torch.eye(3)
        # print(angle, translations, scale, shear, patch_location)

        matrix = torchvision.transforms.functional._get_inverse_affine_matrix([0, 0], angle, translations, scale,
                                                                              shear, )

        return F.affine(img, angle, translations, scale, shear, interpolation=self.interpolation, fill=fill,
                        center=patch_location), matrix, patch_location

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", scale={self.scale}" if self.scale is not None else ""
        s += f", shear={self.shear}" if self.shear is not None else ""
        s += f", interpolation={self.interpolation.value}" if self.interpolation != InterpolationMode.NEAREST else ""
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += f", center={self.center}" if self.center is not None else ""
        s += f", patch_translate={self.patch_translate}" if self.patch_translate is not None else ""

        s += ")"

        return s


def translation_transform(tx,ty):
    transform=np.eye(3)
    transform[0,2]=tx
    transform[1,2]=ty
    return transform


class SARImages (Dataset):
    """ Loads all images from the Aachen Day-Night dataset 
    """
    def __init__(self, select='agri urban', root='data/sar'):
        Dataset.__init__(self)
        self.root = root
        self.img_dir = 's1'
        self.select = set(select.split())
        assert self.select, 'Nothing was selected'
        
        self.imgs = []
        root = os.path.join(root, self.img_dir)
        for dirpath, _, filenames in os.walk(root):
            r = dirpath[len(root)+1:]
            if not(self.select & set(r.split('/'))): continue
            self.imgs += [os.path.join(r,f) for f in filenames if f.endswith('.png')]
        
        self.nimg = len(self.imgs)
        assert self.nimg, 'Empty SAR dataset'

    def get_key(self, idx):
        return self.imgs[idx]



class SARImages_DB (SARImages):
    """ Only database (db) images.
    """
    def __init__(self, **kw):
        SARImages.__init__(self, select='urban agri', **kw)
        self.db_image_idxs = {self.get_tag(i) : i for i,f in enumerate(self.imgs)}
    
    def get_tag(self, idx): 
        # returns image tag == img number (name)
        return os.path.split( self.imgs[idx][:-4] )[1]


class SAR_OpticalFlow( PairDataset):
    """ Image pairs from Aachen db with optical flow.
    """
    def __init__(self, root='data/sar/optical_flow', **kw):
        PairDataset.__init__(self)
        # SARImages_DB.__init__(self, **kw)
        self.root_flow = root
        self.random_affine = RandomAffine(degrees=10, translate=(.05, .05), scale=(.8, 1), shear=(.1, .1), patch_translate=((.2, 0.8), (.2, 0.8)))
        self.patch_radius = 128
        # find out the subsest of valid pairs from the list of flow files
        self.nimg = 0
        all_image_sets = []
        for set_img  in os.listdir(root):
            for path, _, files in os.walk(os.path.join(root, set_img)):
                file_path = [os.path.join(path, f) for f in files if f[-3:].lower() in ["png", "jpg"]]
                self.nimg += 1
                if len(file_path)>0 :
                    all_image_sets.append(file_path)

        self.data_base = all_image_sets
        self.npairs = len(all_image_sets)
        vector = np.arange(0 , 2*self.patch_radius)
        n , m =  np.meshgrid(vector, vector)
        self.locs = np.c_[n.flatten(), m.flatten()]



    def create_optical_flow(self, transformation):
        new_locs = transformation.inverse(self.locs)
        aflow = new_locs.reshape((2 * self.patch_radius, 2 * self.patch_radius, 2))
        return aflow

    # def get_pairs(self, idx):
    #
    #     return patch1, patch2, meta

    def get_pair(self, idx, output=()):
        if isinstance(output, str):
            output = output.split()

        assert idx <= len(self.data_base)
        meta = {}
        img_set = self.data_base[idx]
        random.shuffle(img_set)
        img1_path, img2_path = img_set[:2]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img2 = img2.resize(img1.size)
        # img2 = img1
        transformed_img1, affine_matrix, patch_location = self.random_affine(img1)
        tr_mat = torch.Tensor(affine_matrix + [0, 0, 1]).reshape(3, 3)
        final_tr_mat = translation_transform(self.patch_radius, self.patch_radius) @ tr_mat.numpy() @ np.linalg.inv(
            translation_transform(self.patch_radius, self.patch_radius))
        n, m = patch_location
        affine_tr = AffineTransform(matrix=final_tr_mat)
        # img1 = transform.to_tensor(img1)
        img2 = transform.to_tensor(img2)
        transformed_img1 = transform.to_tensor( transformed_img1)
        patch1 = img2[0, m - self.patch_radius:m + self.patch_radius, n - self.patch_radius:n + self.patch_radius]
        patch2 = transformed_img1[ 0, m - self.patch_radius:m + self.patch_radius,
                 n - self.patch_radius:n + self.patch_radius]
        optical_flow = self.create_optical_flow(affine_tr)
        mask = warp(np.ones_like(patch1), affine_tr.inverse, preserve_range=True).astype('bool')

        meta["aflow"] = optical_flow
        meta["mask"] = mask
        patch1 = patch1[None].repeat_interleave(3, 0)
        patch2 = patch2[None].repeat_interleave(3, 0)
        img1 = transform.to_pil_image(patch1)
        img2 = transform.to_pil_image(patch2)
        # np.save('img1', img1)
        # np.save('img2', img2)
        # np.save('mask', meta['mask'])
        # np.save('aflow', meta['aflow'])
        # np.save('transformation', final_tr_mat)

        return img1, img2, meta


if __name__ == '__main__':
    print(sar_db_images)
    print(sar_db_flow)
    pdb.set_trace()
