# Copyright (c) OpenMMLab. All rights reserved.
import logging
import re
from typing import Sequence
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from mmengine.visualization import Visualizer
from scipy import io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mmagic.registry import VISUALIZERS
from mmagic.structures import DataSample
from mmagic.utils import print_colored_log

@VISUALIZERS.register_module()
class ConcatLSTVisualizer(Visualizer):
    """Visualize multiple LSTs by concatenation.

    This visualizer will horizontally concatenate images belongs to different
    keys and vertically concatenate images belongs to different frames to
    visualize.

    Image to be visualized can be:
        - torch.Tensor or np.array
        - Image sequences of shape (T, C, H, W)
        - Multi-channel image of shape (1/3, H, W)
        - Single-channel image of shape (C, H, W)

    Args:
        fn_key (str): key used to determine file name for saving image.
            Usually it is the path of some input image. If the value is
            `dir/basename.ext`, the name used for saving will be basename.
        img_keys (str): keys, values of which are images to visualize.
        pixel_range (dict): min and max pixel value used to denormalize images,
            note that only float array or tensor will be denormalized,
            uint8 arrays are assumed to be unnormalized.
        bgr2rgb (bool): whether to convert the image from BGR to RGB.
        name (str): name of visualizer. Default: 'visualizer'.
        *args and \**kwargs: Other arguments are passed to `Visualizer`. # noqa
    """
    MAPPING = {"input": "LR_LST", "pred_img": "Pred_LST", "gt_img": "GT_LST"}

    def __init__(
        self,
        fn_key: str,  # filename key for saving img.
        img_keys: Sequence[str],
        mask_keys: Sequence[str] = None,
        apply_mask: bool = True,
        name: str = "visualizer",
        use_color_map: bool = False,
        show_imgs: bool = False,
        add_title: bool = False,
        show_diff: bool = False,
        show_color_bar: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name, *args, **kwargs)
        self.fn_key = fn_key
        self.img_keys = img_keys
        self.apply_mask = apply_mask
        self.use_color_map = use_color_map
        self.mask_keys = mask_keys
        self.show_imgs = show_imgs
        self.add_title = add_title
        self.show_diff = show_diff
        self.show_color_bar = show_color_bar

        if self.mask_keys is not None:
            assert len(self.img_keys) == len(self.mask_keys)

    def add_datasample(self, data_sample: DataSample, step=0) -> None:
        """Concatenate image and draw.

        Args:
            input (torch.Tensor): Single input tensor from data_batch.
            data_sample (DataSample): Single data_sample from data_batch.
            output (DataSample): Single prediction output by model.
            step (int): Global step value to record. Default: 0.
        """
        # Note:
        # with LocalVisBackend and default arguments, we have:
        # self.save_dir == runner._log_dir / 'vis_data'

        merged_dict = {
            **data_sample.to_dict(),
        }

        if "output" in merged_dict.keys():
            merged_dict.update(**merged_dict["output"])

        fn = None
        if self.fn_key in merged_dict:
            fn = merged_dict[self.fn_key]
            if isinstance(fn, list):
                fn = fn[0]
            fn = re.split(r" |/|\\", fn)[-1]
            fn = fn.split(".")[0]
        if (fn is None) or (len(str(fn)) == 0):
            fn = f"sample_{step:06d}"
        else:
            fn = f"{fn}_{step:06d}"

        img_list = []
        for k in self.img_keys:
            if k not in merged_dict:
                print_colored_log(f'Key "{k}" not in data_sample or outputs', level=logging.WARN)
                continue

            img = merged_dict[k]

            # PixelData
            if isinstance(img, dict) and ("data" in img):
                img = img["data"]

            # Tensor to array: chw->hwc or tchwd->thwc
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                elif img.ndim == 4:
                    img = img.transpose(0, 2, 3, 1)

            # concat frame vertically or pick first frame for .mat saving
            if img.ndim == 4:
                if self.use_color_map:
                    img = np.concatenate(img, axis=0)
                else:
                    img = img[0]

            img_list.append(img)

        # visualize and save LST images list.
        if self.use_color_map:
            self.vis_save_lst(merged_dict, img_list, fn)
        else:
            self.save_to_mat(img_list, fn)
    
    def save_to_mat(self, img_list: Sequence[np.array], filename: str):
        if len(img_list) == 0:
            print_colored_log('No images to save in img_list.', level=logging.WARN)
            return

        data = img_list[0]

        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                print_colored_log(f'Cannot convert data to numpy array: {e}', level=logging.ERROR)
                return

        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)

        if data.ndim == 2:
            data = data[..., np.newaxis]
        elif data.ndim == 3:
            if data.shape[-1] != 1:
                data = data[..., :1]
        else:
            print_colored_log(f'Unexpected data ndim={data.ndim}, expected 2 or 3.', level=logging.ERROR)
            return

        filename = filename + ".mat"
        save_path = os.path.join(self._vis_backends["LocalVisBackend"]._save_dir, filename)
        sio.savemat(save_path, mdict={"data": data})

    def vis_save_lst(self, merged_dict: dict, img_list: Sequence[np.array], filename: str):
        
        if self.show_diff:
            num_imgs = len(img_list) + 1
        else:
            num_imgs = len(img_list)
        fig, axes = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5))
        # "jet" "hot" "plasma" "magma" "inferno" "cividis" "viridis"
        cmap = plt.get_cmap("viridis")
        if num_imgs == 1:
            axes = [axes]

        for i, data in enumerate(img_list):
            img_key = self.img_keys[i]
            if self.mask_keys is not None:
                mask_key = self.mask_keys[i]
                mask = merged_dict[mask_key].detach().cpu().numpy()
                mask = mask == 1
                mask = np.transpose(mask, (1, 2, 0))
                # we set nan value for data[~mask], we need white background.
                data[~mask] = np.nan

                vmin = np.min(data[mask])
                vmax = np.max(data[mask])
            else:
                vmin = np.min(data)
                vmax = np.max(data)
            ax = axes[i]
            img = axes[i].imshow(data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
            if self.add_title:
                ax.set_title(str(self.MAPPING[img_key]))

            ax.set_frame_on(True)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if self.show_diff:
            pred_img = img_list[1]
            gt_img = img_list[2]
            diff_data = pred_img - gt_img
            if self.mask_keys is not None:
                mask_key = "hr_mask"
                mask = merged_dict[mask_key].detach().cpu().numpy()
                mask = mask == 1
                mask = np.transpose(mask, (1, 2, 0))
                # we set nan value for diff_data[~mask], we need white background.
                diff_data[~mask] = np.nan

                vmin = np.min(diff_data[mask])
                vmax = np.max(diff_data[mask])
            else: 
                vmin = np.min(diff_data)
                vmax = np.max(diff_data)
            # print(f"vmin={vmin}, vmax={vmax}")
            ax = axes[-1]
            ax.set_title("Pred-GT")
            diff_img = axes[-1].imshow(diff_data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
            # ax.set_title("GT-Pred")
            # diff_img = axes[-1].imshow(-diff_data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)

            ax.set_frame_on(True)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # add cbar for diff
            if self.show_color_bar:
                divider = make_axes_locatable(axes[-1])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(diff_img, cax=cax, orientation="vertical")
                cbar.set_label("Temperature/K")

        if self.show_color_bar:
            divider = make_axes_locatable(axes[-2])
            cax = divider.append_axes("right", size="5%", pad=0.1) 
            cbar = plt.colorbar(img, cax=cax, orientation="vertical")
            cbar.set_label("Temperature/K")

        # add title
        if self.add_title:
            plt.suptitle(filename)
        filename = filename + ".png"
        save_path = os.path.join(self._vis_backends["LocalVisBackend"]._save_dir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False, pad_inches=0)

        if self.show_imgs:
            plt.show()
            plt.pause(1)
        plt.close()
