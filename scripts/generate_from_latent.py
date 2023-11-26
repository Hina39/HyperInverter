from utils import common
import argparse
import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from models.hyper_inverter import HyperInverter  # noqa: E402
from options.test_options import TestOptions  # noqa: E402
from PIL import Image  # noqa: E402
from utils.common import tensor2im, unflatten_dict  # noqa: E402


def run():
    # 出力ディレクトリを設定
    test_opts = TestOptions().parse()
    out_path_results = os.path.join(test_opts.exp_dir, "inference_results")
    out_path_coupled = os.path.join(test_opts.exp_dir, "inference_coupled")

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # 保存されたモデル（HyperInverter）を読み込み →学習済みの重みを読み込む
    ckpt = torch.load(test_opts.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    opts = argparse.Namespace(**opts)
    net = HyperInverter(opts)
    net.eval()
    net.cuda()

    # npzファイルを読み込む
    npz_path = test_opts.npz_path
    latents = unflatten_dict(np.load(npz_path), separator="-")
    predicted_weights = latents["predicted_weights"]
    w_codes = latents["w_codes"]

    predicted_weights = {
        key: torch.from_numpy(value).to("cuda")
        for key, value in predicted_weights.items()
    }
    w_codes = torch.from_numpy(w_codes).to("cuda")

    with torch.no_grad():
        w_images = net.decoder[0].synthesis(
            w_codes, added_weights=None, noise_mode="const"
        )

    print(w_images.shape)  # torch.Size([1, 3, 1024, 1024])

    pred_weights_per_sample = {}
    for key in predicted_weights:
        pred_weights_per_sample[key] = predicted_weights[key][0]

    added_weights = common.convert_predicted_weights_to_dict(pred_weights_per_sample)

    final_image = (
        net.decoder[0]
        .synthesis(w_codes, added_weights=added_weights, noise_mode="const")
        .squeeze(0)
    )

    print(final_image.shape)

    # 画像を保存する.
    _w_image = tensor2im(w_images[0])
    _final_image = tensor2im(final_image)

    res = np.concatenate(
        [np.array(_w_image), np.array(_final_image)],
        axis=1,
    )
    Image.fromarray(res).save(
        os.path.join(out_path_coupled, os.path.basename(npz_path.stem) + ".png")
    )

    im_save_path = os.path.join(
        out_path_results, os.path.basename(npz_path.stem) + ".png"
    )
    Image.fromarray(np.array(_final_image)).save(im_save_path)


if __name__ == "__main__":
    run()
