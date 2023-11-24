# npzからlatentsを読み出して, 画像を生成するスクリプトを実装


import argparse
import os
import sys


sys.path.append(".")
sys.path.append("..")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from configs import data_configs  # noqa: E402
from datasets.inference_dataset import InferenceDataset  # noqa: E402
from models.hyper_inverter import HyperInverter  # noqa: E402
from options.test_options import TestOptions  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.common import log_input_image, tensor2im, unflatten_dict  # noqa: E402
from utils import common


def run():
    # npzからlatentsを読み出す
    # predicted_weights, w_codes = np.load("/home/challenger/hyperinverter/outputs2/latents/10380049_598442736930128_1821329686642729572_o.npz")
    # print(predicted_weights.shape)
    # print(w_codes.shape)

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

    # データセットとデータローダーを設定
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

    # データローダーからバッチを取得し、それぞれのバッチに対して以下の操作を行います：
    # バッチをGPUに移動し、float型に変換します。
    # ネットワークを用いて画像を生成します。
    # 生成した画像を保存します。
    global_i = 0
    global_time = 0
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        # with torch.no_grad():
        #     input_cuda = input_batch.cuda().float()
        #     tic = time.time()
        #     w_images, final_images, predicted_weights = run_on_batch(input_cuda, net)
        #     toc = time.time()
        #     global_time += toc - tic

        npz_file = np.load(
            "/home/challenger/hyperinverter/outputs2/latents/10380049_598442736930128_1821329686642729572_o.npz"
        )
        npz_file = unflatten_dict(npz_file, separator="-")
        print([key for key in npz_file.keys()])
        final_images, w_images = run_on_batch(npz_file, net)
        # npz_fileこ
        bs = 2
        for i in range(bs):
            final_image = tensor2im(final_images[i])
            w_images = tensor2im(w_images[i])

            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i])
                res = np.concatenate(
                    [np.array(input_im), np.array(final_image)],
                    axis=1,
                )
                Image.fromarray(res).save(
                    os.path.join(out_path_coupled, os.path.basename(im_path))
                )

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(final_image)).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, "stats.txt")
    result_str = "Runtime {:.4f}".format(global_time / len(dataset))
    print(result_str)

    with open(stats_path, "w") as f:
        f.write(result_str)

    # predicted_weights, w_codes = np.load("/home/challenger/hyperinverter/outputs2/latents/10380049_598442736930128_1821329686642729572_o.npz")


def run_on_batch(npz_file, net):
    predicted_weights = npz_file["predicted_weights"]
    w_codes = npz_file["w_codes"]

    predicted_weights = {
        key: torch.from_numpy(value).to("cuda")
        for key, value in predicted_weights.items()
    }
    w_codes = torch.from_numpy(w_codes).to("cuda")

    with torch.no_grad():
            w_images = net.decoder[0].synthesis(
                w_codes, added_weights=None, noise_mode="const"
            )

    # forwardもよばれる　net()はHyperInverterクラスの__call__メソッドを呼び出している

    # Generate final images from predicted weights and w codes
    final_images = []
    bs = 2
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

        final_image = (
            net.decoder[idx]
            .synthesis(w_codes, added_weights=added_weights, noise_mode="const")
            .squeeze(0)
        )
    print(final_image)
    final_images.append(final_image)

    final_images = torch.stack(final_images, 0)

    print(final_images.shape)
    print(type(final_image))

    return final_images, w_images

    return_data = [w_images, final_images, predicted_weights]
    if return_latents:
        return_data.append(w_codes)

    return tuple(return_data)


if __name__ == "__main__":
    run()
