'''
Author: Jiaxin Zheng
Date: 2023-09-03 11:03:18
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:54:57
Description: 
'''
import math
import random
from typing import Any
import numpy as np
import cv2
import albumentations as A
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.functional import safe_rotate_enlarged_img_size, _maybe_process_in_chunks, \
                                                              keypoint_rotate
from albumentations.augmentations.geometric.transforms import Affine
import torch

cv2.setNumThreads(1)

def box_cxcywh_to_xyxy(x_c, y_c, w, h):
    return (x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)

def box_xyxy_to_cxcywh(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)

def merge_bbox_keypoints(coords,bbox_list,thr=3):
    extended_coords = []
    for i,(coord,bbox) in enumerate(zip(coords,bbox_list)):
        if bbox[2] - bbox[0] > thr and bbox[3] - bbox[1] > thr:
            extended_coords.append(bbox.tolist())
        else:
            t_coord = [i+1e-6 for i in coord.tolist()]
            extended_coords.append(coord.tolist()+t_coord)
    return extended_coords

def merge_bbox_keypoints_expand(coords,bbox_list,thr=3):
    extended_coords = []
    for i,(coord,bbox) in enumerate(zip(coords,bbox_list)):
        if bbox[2] - bbox[0] > thr and bbox[3] - bbox[1] > thr:
            # extended_coords.append(bbox.tolist())
            extended_coords.append(bbox.tolist()[:2])
            extended_coords.append(bbox.tolist()[2:])
        else:
            t_coord = [i+1e-6 for i in coord.tolist()]
            # extended_coords.append(coord.tolist()+t_coord)
            extended_coords.append(coord.tolist())
            extended_coords.append(coord.tolist())
    return extended_coords

def get_bbox_list_from_s_list(coords):
    coords_bbox = []
    for i in list(range(0,len(coords),2)):
        coords_bbox.append(coords[i].tolist()+coords[i+1].tolist())
    return coords_bbox

def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)

def safe_rotate(
    img: np.ndarray,
    angle: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    value: int = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
):

    old_rows, old_cols = img.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (old_cols / 2, old_rows / 2)

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    # Rotation Matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Shift the image to create padding
    rotation_mat[0, 2] += new_cols / 2 - image_center[0]
    rotation_mat[1, 2] += new_rows / 2 - image_center[1]

    # CV2 Transformation function
    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=rotation_mat,
        dsize=(new_cols, new_rows),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    # rotate image with the new bounds
    rotated_img = warp_affine_fn(img)

    return rotated_img


def keypoint_safe_rotate(keypoint, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = (new_cols - old_cols) / 2
    row_diff = (new_rows - old_rows) / 2

    # Shift keypoint
    shifted_keypoint = (int(keypoint[0] + col_diff), int(keypoint[1] + row_diff), keypoint[2], keypoint[3])

    # Rotate keypoint
    rotated_keypoint = keypoint_rotate(shifted_keypoint, angle, rows=new_rows, cols=new_cols)

    return rotated_keypoint


class SafeRotate(A.SafeRotate):

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(SafeRotate, self).__init__(
            limit=limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            p=p)

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return safe_rotate(
            img=img, value=self.value, angle=angle, interpolation=interpolation, border_mode=self.border_mode)

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return keypoint_safe_rotate(keypoint, angle=angle, rows=params["rows"], cols=params["cols"])


class CropWhite(A.DualTransform):
    
    def __init__(self, value=(255, 255, 255), pad=0, p=1.0):
        super(CropWhite, self).__init__(p=p)
        self.value = value
        self.pad = pad
        # assert pad >= 0

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        x = (img != self.value).sum(axis=2)
        if x.sum() == 0:
            return params
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top+1 < height:
            top += 1
        bottom = height
        while row_sum[bottom-1] == 0 and bottom-1 > top:
            bottom -= 1
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left+1 < width:
            left += 1
        right = width
        while col_sum[right-1] == 0 and right-1 > left:
            right -= 1
        # crop_top = max(0, top - self.pad)
        # crop_bottom = max(0, height - bottom - self.pad)
        # crop_left = max(0, left - self.pad)
        # crop_right = max(0, width - right - self.pad)
        # params.update({"crop_top": crop_top, "crop_bottom": crop_bottom,
        #                "crop_left": crop_left, "crop_right": crop_right})
        params.update({"crop_top": top, "crop_bottom": height - bottom,
                       "crop_left": left, "crop_right": width - right})
        return params

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width, _ = img.shape
        w_pad, h_pad = int(width * 0.15), int(height * 0.15)
        
        img = img[crop_top:height - crop_bottom, crop_left:width - crop_right]
        
        if self.pad>=0:
            img = A.augmentations.pad_with_params(
                img, self.pad, self.pad, self.pad, self.pad, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        else:
            # print(f'b:{img.shape}')
            img = A.augmentations.pad_with_params(
            img, h_pad, h_pad, w_pad, w_pad, border_mode=cv2.BORDER_CONSTANT, value=self.value)
            self.h_pad = h_pad
            self.w_pad = w_pad
            # print(f'a:{img.shape}')
        return img

    def apply_to_keypoint(self, keypoint, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        if self.pad>=0:
            return x - crop_left + self.pad, y - crop_top + self.pad, angle, scale
        else:
            return x - crop_left + self.w_pad, y - crop_top + self.h_pad, angle, scale

    def apply_to_bbox(self,
        bbox ,
        rows: int = 0,
        cols: int = 0,
        crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
        **params
    ):
        x1, y1, x2, y2 = denormalize_bbox(bbox, rows, cols)
        if self.pad>=0:
            x1, y1, x2, y2 = x1 - crop_left + self.pad, y1 - crop_top + self.pad, x2 - crop_left + self.pad, y2 - crop_top + self.pad
        else:
            x1, y1, x2, y2 =  x1 - crop_left + self.w_pad, y1 - crop_top + self.h_pad, x2 - crop_left + self.w_pad, y2 - crop_top + self.h_pad
        
        bbox = normalize_bbox((x1, y1, x2, y2), rows, cols)
        
        return bbox
        
    def get_transform_init_args_names(self):
        return ('value', 'pad')


class PadWhite(A.DualTransform):

    def __init__(self, pad_ratio=0.2, p=0.5, value=(255, 255, 255)):
        super(PadWhite, self).__init__(p=p)
        self.pad_ratio = pad_ratio
        self.value = value

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        side = random.randrange(4)
        if side == 0:
            params['pad_top'] = int(height * self.pad_ratio * random.random())
        elif side == 1:
            params['pad_bottom'] = int(height * self.pad_ratio * random.random())
        elif side == 2:
            params['pad_left'] = int(width * self.pad_ratio * random.random())
        elif side == 3:
            params['pad_right'] = int(width * self.pad_ratio * random.random())
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        height, width, _ = img.shape
        img = A.augmentations.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img
    
    def apply_to_bbox(self,
        bbox ,
        rows: int = 0,
        cols: int = 0,
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
        **params
    ):
        x1, y1, x2, y2 = denormalize_bbox(bbox, rows, cols)
        x1, y1, x2, y2 = x1 + pad_left, y1 + pad_top, x2 + pad_right, y2 + pad_bottom
        
        bbox = normalize_bbox((x1, y1, x2, y2), rows, cols)
        
        return bbox
    
    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return ('value', 'pad_ratio')


class SaltAndPepperNoise(A.DualTransform):

    def __init__(self, num_dots, value=(0, 0, 0), p=0.5):
        super().__init__(p)
        self.num_dots = num_dots
        self.value = value

    def apply(self, img, **params):
        height, width, _ = img.shape
        num_dots = random.randrange(self.num_dots + 1)
        for i in range(num_dots):
            x = random.randrange(height)
            y = random.randrange(width)
            img[x, y] = self.value
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox
    
    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ('value', 'num_dots')
    
class ResizePad(A.DualTransform):

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, value=(255, 255, 255)):
        super(ResizePad, self).__init__(always_apply=True)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.value = value

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w, _ = img.shape
        img = A.augmentations.geometric.functional.resize(
            img, 
            height=min(h, self.height), 
            width=min(w, self.width), 
            interpolation=interpolation
        )
        h, w, _ = img.shape
        pad_top = (self.height - h) // 2
        pad_bottom = (self.height - h) - pad_top
        pad_left = (self.width - w) // 2
        pad_right = (self.width - w) - pad_left
        img = A.augmentations.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=cv2.BORDER_CONSTANT,
            value=self.value,
        )
        return img


def normalized_grid_distortion(
        img,
        num_steps=10,
        xsteps=(),
        ysteps=(),
        *args,
        **kwargs
):
    height, width = img.shape[:2]

    # compensate for smaller last steps in source image.
    x_step = width // num_steps
    last_x_step = min(width, ((num_steps + 1) * x_step)) - (num_steps * x_step)
    xsteps[-1] *= last_x_step / x_step

    y_step = height // num_steps
    last_y_step = min(height, ((num_steps + 1) * y_step)) - (num_steps * y_step)
    ysteps[-1] *= last_y_step / y_step

    # now normalize such that distortion never leaves image bounds.
    tx = width / math.floor(width / num_steps)
    ty = height / math.floor(height / num_steps)
    xsteps = np.array(xsteps) * (tx / np.sum(xsteps))
    ysteps = np.array(ysteps) * (ty / np.sum(ysteps))

    # do actual distortion.
    return A.augmentations.functional.grid_distortion(img, num_steps, xsteps, ysteps, *args, **kwargs)


class NormalizedGridDistortion(A.augmentations.transforms.GridDistortion):
    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        return normalized_grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode,
                                          self.value)

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        return normalized_grid_distortion(
            img, self.num_steps, stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)


def get_transforms(input_size, augment=True, rotate=True, debug=False,deformation=False):
    trans_list = []
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    # trans_list.append(CropWhite(pad=5))
    trans_list.append(CropWhite(pad=-1))
    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            # PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    if not deformation:
        trans_list += [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(
                min_height=input_size, min_width=input_size, value=(255, 255, 255),border_mode=cv2.BORDER_CONSTANT
            ),
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list,keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_transforms_hard(input_size, augment=True, rotate=True, debug=False,deformation=False,interpolation=4):
    """
        :param: interpolation:(OpenCV flag)
                0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_CUBIC, 3: cv2.INTER_AREA, 4: cv2.INTER_LANCZOS4.
                Default: cv2.INTER_LINEAR. Current Default: cv2.INTER_LANCZOS4
    """
    trans_list = []
    
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        trans_list.append(Affine(
                    shear=({"x": (-30, 30), "y": (-30, 30)}),
                    p=1,
                    fit_output=True,
                    cval=(255, 255, 255)))
    trans_list.append(CropWhite(pad=-1))
    
    if augment:
        trans_list += [
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    if not deformation:
        trans_list += [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(
                min_height=input_size, min_width=input_size, value=(255, 255, 255),border_mode=cv2.BORDER_CONSTANT
            ),
        ]
    trans_list.append(A.Resize(input_size, input_size,interpolation=interpolation))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),  # CHW
        ]
    else:
        trans_list += [
            A.ToGray(p=1),
            ToTensorV2()]
    return A.Compose(trans_list,keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))