# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from monai.utils.type_conversion import convert_to_numpy


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for TensorBoard visualization.
    Args:
        image: image tensor or numpy array, sized (H, W, D)
    Returns:
        torch.Tensor: formatted for TensorBoard (C, H, W) with values in [0, 1]
    """
    try:
        # Convert to numpy first
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = convert_to_numpy(image)
        
        # Handle different array shapes
        if image_np.ndim != 3:
            return None
            
        # Draw image slice
        center = image_np.shape[axis] // 2
        if axis == 0:
            draw_img = image_np[center, :, :]
        elif axis == 1:
            draw_img = image_np[:, center, :]
        elif axis == 2:
            draw_img = image_np[:, :, max(0, center-10)]
        else:
            return None
        
        # Normalize to [0, 1]
        if np.amin(draw_img) < 0:
            draw_img = draw_img - np.amin(draw_img)
        if np.amax(draw_img) > 0:
            draw_img = draw_img / np.amax(draw_img)
        
        # Ensure the image is 2D
        if draw_img.ndim != 2:
            return None
            
        # Convert to tensor with proper format for TensorBoard (C, H, W)
        # TensorBoard expects (C, H, W) format where C=1 for grayscale
        draw_tensor = torch.from_numpy(draw_img).unsqueeze(0).float()
        
        return draw_tensor
        
    except Exception as e:
        print(f"Error in visualize_one_slice_in_3d_image: {e}")
        return None
