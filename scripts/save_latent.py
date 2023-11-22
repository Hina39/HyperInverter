"""

Inverterを用いて潜在変数を推定し, npzファイルとして保存する.
このスクリプトはscripts/inference.pyを参考に最低限の変更を加えたものです.

"""
import argparse
import os
import sys
import time
import pathlib

sys.path.append(".")
sys.path.append("..")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from configs import data_configs  # noqa: E402
from datasets.inference_dataset import InferenceDataset  # noqa: E402
from models.hyper_inverter import HyperInverter  # noqa: E402
from options.test_options import TestOptions  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.common import flatten_dict  # noqa: E402


def run() -> None:
    """潜在変数をnpzファイルとして保存する.

    保存先は, <OUTPUT_DIR_PATH>/latents/以下になります. ファイル名が画像名と同じ
    になります.

    % poetry run python scripts/save_latent.py \
        --exp_dir <OUTPUT_DIR_PATH> \
        --checkpoint_path pretrained_models/<PRETRAINED_WEIGHT> \
        --data_path <DATA_DIR_PATH>" \
        --batch_size 1 \
        --workers 4

    """
    test_opts = TestOptions().parse()

    # latentsの保存先のディレクトリを作成.
    out_path_latents = pathlib.Path(test_opts.exp_dir, "latents")
    out_path_latents.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(test_opts.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    opts = argparse.Namespace(**opts)

    # オプションを表示.
    print("Options:")
    print(opts)

    net = HyperInverter(opts)
    net.eval()
    net.cuda()

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args["transforms"](opts).get_transforms()
    dataset = InferenceDataset(
        root=opts.data_path, transform=transforms_dict["transform_inference"], opts=opts
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=int(opts.workers),
        drop_last=False,
    )

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = 0
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            # ticは処理の開始時刻.
            tic = time.time()
            # retun_latents=Trueにしたので, w_codesも返ってくる.
            _, _, predicted_weights, w_codes = run_on_batch(input_cuda, net)
            # tocは処理の終了時刻.
            toc = time.time()
            global_time += toc - tic

            # 保存するためにnumpyに変換.
            _predicted_weights = {
                key: value.cpu().numpy() for key, value in predicted_weights.items()
            }
            _w_codes = w_codes.cpu().numpy()
            latents = {
                "predicted_weights": _predicted_weights,
                "w_codes": _w_codes,
            }

        # bsはbatch sizeの略.
        # w_codesの最初のチャネルはバッチサイズになっている.
        bs = w_codes.size(0)
        for _ in range(bs):
            # npz形式で保存.
            save_path_predicted_latents = out_path_latents / (
                pathlib.Path(dataset.paths[global_i]).stem + ".npz"
            )
            np.savez(
                str(save_path_predicted_latents), **flatten_dict(latents, separator="-")
            )
            global_i += 1

    stats_path = os.path.join(opts.exp_dir, "stats.txt")
    result_str = "Runtime {:.4f}".format(global_time / len(dataset))
    print(result_str)

    with open(stats_path, "w") as f:
        f.write(result_str)


def run_on_batch(inputs, net):
    # w_codesの情報も欲しいので、return_latents=Trueにする.
    result_batch = net(inputs, return_latents=True)
    return result_batch


if __name__ == "__main__":
    run()
