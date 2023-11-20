import argparse
import copy
import math
import pickle

import numpy as np
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from models.encoders import fpn_encoders
from models.hypernetwork import Hypernetwork
from models.stylegan2_ada import Discriminator, Generator
from models.weight_shapes import (
    STYLEGAN2_ADA_ALL_WEIGHT_WITHOUT_BIAS_SHAPES,
    STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_SHAPES,
    STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_WITHOUT_TO_RGB_SHAPES,
)
from utils import common
from utils.model_utils import RESNET_MAPPING


def get_target_shapes(opts):
    general_shape = None  # Add this line
    if opts.target_shape_name == "conv_without_bias":
        general_shape = STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_SHAPES
    elif opts.target_shape_name == "all_without_bias":
        general_shape = STYLEGAN2_ADA_ALL_WEIGHT_WITHOUT_BIAS_SHAPES
    elif opts.target_shape_name == "conv_without_bias_without_torgb":
        general_shape = STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_WITHOUT_TO_RGB_SHAPES

    # Add a check to ensure general_shape is not None
    if general_shape is None:
        raise ValueError(f"Invalid target_shape_name: {opts.target_shape_name}")

    target_shape = {}
    for layer_name in general_shape:
        cur_resolution = int(layer_name.split(".")[0][1:])
        if cur_resolution <= opts.output_size:
            target_shape[layer_name] = general_shape[layer_name]

    return target_shape


class HyperInverter(nn.Module):
    def __init__(self, opts):
        super().__init__()

        # Configurations
        self.set_opts(opts)

        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2

        # Hypernetwork
        self.target_shape = get_target_shapes(self.opts)
        self.hypernet = Hypernetwork(
            input_dim=512,
            hidden_dim=self.opts.hidden_dim,
            target_shape=self.target_shape,
        )

        # Define and load architecture
        self.load_weights()

        # For visualization
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def load_weights(self):
        # Load W-Encoder (E1 Encoder in Paper)
        if self.opts.w_encoder_path is not None:
            w_encoder_path = self.opts.w_encoder_path
        elif (
            self.opts.dataset_type == "ffhq_encode"
            and model_paths["w_encoder_ffhq"] is not None
        ):
            w_encoder_path = model_paths["w_encoder_ffhq"]
        elif (
            self.opts.dataset_type == "church_encode"
            and model_paths["w_encoder_church"] is not None
        ):
            w_encoder_path = model_paths["w_encoder_church"]
        else:
            raise Exception("Please specify the path to the pretrained W encoder.")

        print(f"Loaded pretrained W encoder from: {w_encoder_path}")

        ckpt = torch.load(w_encoder_path, map_location="cpu")

        opts = ckpt["opts"]
        opts = argparse.Namespace(**opts)

        if "ffhq" in self.opts.dataset_type or "celeb" in self.opts.dataset_type:
            # Using ResNet-IRSE50 for facial domain
            self.w_encoder = fpn_encoders.BackboneEncoderUsingLastLayerIntoW(
                50, "ir_se", opts
            )
        else:
            # Using ResNet34 pre-trained on ImageNet for other domains
            self.w_encoder = fpn_encoders.ResNetEncoderUsingLastLayerIntoW()

        self.w_encoder.load_state_dict(common.get_keys(ckpt, "encoder"), strict=True)
        self.w_encoder.to(self.opts.device).eval()
        common.toogle_grad(self.w_encoder, False)

        # Load pretrained StyleGAN2-ADA models
        if self.opts.dataset_type == "ffhq_encode":
            stylegan_ckpt_path = model_paths["stylegan2_ada_ffhq"]
        elif self.opts.dataset_type == "church_encode":
            stylegan_ckpt_path = model_paths["stylegan2_ada_church"]

        D_original = None  # Add this line
        with open(stylegan_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

            # Generator
            G_original = ckpt["G_ema"]
            G_original = G_original.float()

            # Load discriminator if we use adversarial loss
            if self.opts.hyper_adv_lambda > 0:
                # Discriminator
                D_original = ckpt["D"]
                D_original = D_original.float()

        # Add a check to ensure D_original is not None
        if D_original is None:
            raise ValueError(
                "D_original is not defined. Check if self.opts.hyper_adv_lambda > 0"
            )

        decoder = Generator(**G_original.init_kwargs)
        decoder.load_state_dict(G_original.state_dict())
        decoder.to(self.opts.device).eval()
        self.decoder = []
        for i in range(self.opts.batch_size):
            self.decoder.append(copy.deepcopy(decoder))

        # Load well-trained discriminator from StyleGAN2 for using adversarial loss
        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.opts.device)

        # Load latent average
        self.latent_avg = self.decoder[0].mapping.w_avg

        # Define W-bar Encoder (E2 Encoder in Paper)
        if self.opts.encoder_type == "LayerWiseEncoder":
            self.w_bar_encoder = fpn_encoders.LayerWiseEncoder(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "ResNetLayerWiseEncoder":
            self.w_bar_encoder = fpn_encoders.ResNetLayerWiseEncoder(self.opts)
        else:
            raise Exception(f"{self.opts.encoder_type} encoder is not defined.")

        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")

            # Load w bar encoder
            self.w_bar_encoder.load_state_dict(
                common.get_keys(ckpt, "w_bar_encoder"), strict=True
            )
            self.w_bar_encoder.to(self.opts.device)

            # Load hypernet
            self.hypernet.load_state_dict(
                common.get_keys(ckpt, "hypernet"), strict=True
            )
            self.hypernet.to(self.opts.device)

            # Load discriminator
            self.discriminator.load_state_dict(
                common.get_keys(ckpt, "discriminator"), strict=True
            )
            self.discriminator.to(self.opts.device)

            print(
                "Loaded pretrained HyperInverter from: {}".format(
                    self.opts.checkpoint_path
                )
            )
        else:
            w_bar_encoder_ckpt = self.__get_encoder_checkpoint()
            self.w_bar_encoder.load_state_dict(w_bar_encoder_ckpt, strict=False)

    def __get_encoder_checkpoint(self):
        if "ffhq" in self.opts.dataset_type:
            print("Loading encoders weights from irse50!")
            encoder_ckpt = torch.load(model_paths["ir_se50"])
            return encoder_ckpt
        else:
            print("Loading encoders weights from resnet34!")
            encoder_ckpt = torch.load(model_paths["resnet34"])
            mapped_encoder_ckpt = dict(encoder_ckpt)
            for p, v in encoder_ckpt.items():
                for original_name, psp_name in RESNET_MAPPING.items():
                    if original_name in p:
                        mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                        mapped_encoder_ckpt.pop(p)
            return encoder_ckpt

    def forward(self, x, return_latents=False):
        bs, _, _, _ = x.size()
        num_ws = self.decoder[0].mapping.num_ws

        # Resize image to feed to encoder
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        # ======== Phase 1 ======== #

        # Obtain w code via W Encoder
        # content code = w_codes
        # E1 = self.w_encoder
        w_codes = self.w_encoder(x)  # bs x 1 x 512

        # Normalize with respect to the center of an average face
        w_codes = w_codes + self.latent_avg.repeat(w_codes.shape[0], 1)
        w_codes = w_codes.unsqueeze(1).repeat([1, num_ws, 1])

        # Genenerate W-images
        # \hat{x}_w = w_images
        # G(w, theta) = self.decoder[0].synthesis
        # delta_theta = added_weights
        with torch.no_grad():
            w_images = self.decoder[0].synthesis(
                w_codes, added_weights=None, noise_mode="const"
            )

        # ======== Phase 2 ======== #

        # Get w_bar code via W bar encoder
        # h_{x} = w_bar_codes
        # E2 = self.w_bar_encoder
        w_bar_codes = self.w_bar_encoder(x)

        # Get w image features
        # 画像のリサイズ
        # \hat{x}_{w} = w_images_resized
        w_images_resized = F.interpolate(
            w_images, size=(256, 256), mode="bilinear", align_corners=False
        )
        # h_{\hat{x}_{w}} = w_image_codes
        # E2 = self.w_bar_encoder
        w_image_codes = self.w_bar_encoder(w_images_resized)

        # Predict weights added to weights of StyleGAN2-Ada synthesis network
        # これが多分 delta_theta = residual weights
        # h = self.hypernet
        predicted_weights = self.hypernet(w_image_codes, w_bar_codes)

        # print([key for key in predicted_weights.keys()])
        print(type(predicted_weights["b4.conv1.weight"]))
        # ['b4.conv1.weight', 'b4.torgb.weight', 'b8.conv0.weight', 'b8.conv1.weight', 'b8.torgb.weight', 'b16.conv0.weight', 'b16.conv1.weight', 'b16.torgb.weight', 'b32.conv0.weight', 'b32.conv1.weight', 'b32.torgb.weight', 'b64.conv0.weight', 'b64.conv1.weight', 'b64.torgb.weight', 'b128.conv0.weight', 'b128.conv1.weight', 'b128.torgb.weight', 'b256.conv0.weight', 'b256.conv1.weight', 'b256.torgb.weight', 'b512.conv0.weight', 'b512.conv1.weight', 'b512.torgb.weight', 'b1024.conv0.weight', 'b1024.conv1.weight', 'b1024.torgb.weight']
        # print(type(w_codes))

        iremono = pathlib.Path("embedding")
        iremono.mkdir(exist_ok=True, parents=True)
        np.savez(
            str(iremono / "kari.npz"),
            predicted_weights=predicted_weights,
            w_codes=w_codes.cpu().numpy(),
        )

        # Generate final images from predicted weights and w codes
        final_images = []

        for idx in range(bs):
            # Add predicted weights to original StyleGAN2-Ada weights
            pred_weights_per_sample = {}
            for key in predicted_weights:
                pred_weights_per_sample[key] = predicted_weights[key][idx]

            # Convert to dict in order to feed to generator
            # これも delta_theta
            # predicted_weights を辞書にしたのだ
            added_weights = common.convert_predicted_weights_to_dict(
                pred_weights_per_sample
            )

            # Gen final image
            w_code = w_codes[idx].unsqueeze(0)
            # \hat{x} = final_image
            # G(w, \hat{theta}) = self.decoder[idx].synthesis
            # \hat{theta} = theta + added_weights(= delta_theta)
            final_image = (
                self.decoder[idx]
                .synthesis(w_code, added_weights=added_weights, noise_mode="const")
                .squeeze(0)
            )
            final_images.append(final_image)

        final_images = torch.stack(final_images, 0)

        return_data = [w_images, final_images, predicted_weights]
        if return_latents:
            return_data.append(w_codes)

        return tuple(return_data)

    def set_opts(self, opts):
        self.opts = opts


if __name__ == "__main__":
    from options.test_options import TestOptions  # noqa: E402
    from PIL import Image

    test_opts = TestOptions().parse()
    # out_path_results = os.path.join(test_opts.exp_dir, "inference_results")
    # out_path_coupled = os.path.join(test_opts.exp_dir, "inference_coupled")

    # os.makedirs(out_path_results, exist_ok=True)
    # os.makedirs(out_path_coupled, exist_ok=True)

    ckpt = torch.load(test_opts.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    opts = argparse.Namespace(**opts)

    hyperinverter = HyperInverter(opts)
    hyperinverter.load_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(
        "/home/challenger/hyperinverter/kiritoridata/10380049_598442736930128_1821329686642729572_o.jpg"
    ).convert("RGB")
    img = torch.from_numpy(np.array(img)).unsqueeze(0).float().permute(0, 3, 1, 2)
    img = img.to(device)
    # print(type(img))
    # print(img.size)
    hyperinverter.forward(img, return_latents=False)
