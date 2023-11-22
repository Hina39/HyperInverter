from PIL import Image
from typing import Dict, MutableMapping, MutableSequence


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


def convert_predicted_weights_to_dict(pred_weights_per_sample):
    """
    Convert data like "conv1.affine.weight : value" to
    {
            "conv1": {
                    "affine": {
                            "weight": value
                    }
            }
            "torgb" : {
                    ...
            }
            ...
    }
    #
    """
    added_weights = {}
    for key in pred_weights_per_sample:
        cur = added_weights
        attr_names = key.split(".")
        for i, attr_name in enumerate(attr_names):
            if i == len(attr_names) - 1:
                cur[attr_name] = pred_weights_per_sample[key]
            elif attr_name not in cur:
                cur[attr_name] = {}
            cur = cur[attr_name]
    return added_weights


# Log images
def log_input_image(x):
    return tensor2im(x)


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def flatten_dict(
    target_dict: MutableMapping,
    separator: str = ".",
) -> Dict:
    """マッピング型の変数を受け取り, その変数に含まれる全ての辞書を平坦化して返す.

    Args:
        target_dict (MutableMapping): 平坦化したいマッピング型の変数
        separator (str): 平坦化した辞書のキーの区切り文字

    Returns:
        Dict: 平坦化した辞書

    Examples:
        >>> config = {
        ...     "classifier": {
        ...         "encoder_base": {
        ...             "_target_": "timm.create_model",
        ...             "model_name": "efficientnet_b4",
        ...             "pretrained": False,
        ...             "num_classes": 1,
        ...         }
        ...     },
        ...     "dataset": {
        ...         "data_module": {
        ...             "_target_": "src.dataset.cifar10.Cifar10DataModule",
        ...             "dataset_stats": {
        ...                 "num_classes": 10,
        ...                 "image_size": {
        ...                     "_target_": "src.dataset.base.ImageSize",
        ...                     "height": 32,
        ...                     "width": 32,
        ...                 },
        ...                 "mean": {
        ...                     "_target_": "src.dataset.base.RgbColor",
        ...                     "r": 0.49139968,
        ...                     "g": 0.48215841,
        ...                     "b": 0.44653091,
        ...                 },
        ...                 "std": {
        ...                     "_target_": "src.dataset.base.RgbColor",
        ...                     "r": 0.24703223,
        ...                     "g": 0.24348513,
        ...                     "b": 0.26158784,
        ...                 },
        ...             },
        ...             "data_root_dir": "./data",
        ...             "batch_size": 128,
        ...             "num_workers": 2,
        ...         },
        ...     },
        ...     "optimizer": {
        ...         "_target_": "torch.optim.SGD",
        ...         "lr": 0.1,
        ...         "momentum": 0.9,
        ...         "weight_decay": 0.0005,
        ...         "dampening": 0.0005,
        ...         "nesterov": False,
        ...     },
        ... }
        >>> flattened_dict = flatten_dict_for_human(cfg)
        >>> for k, v in flattened_dict.items():
        ...     print(f"{k}: {v}")
        ...
        classifier.encoder_base: timm.create_model
        classifier.encoder_base.model_name: efficientnet_b4
        classifier.encoder_base.pretrained: False
        classifier.encoder_base.num_classes: 1
        dataset.data_module: src.dataset.cifar10.Cifar10DataModule
        dataset.data_module.dataset_stats.num_classes: 10
        dataset.data_module.dataset_stats.image_size: src.dataset.base.ImageSize
        dataset.data_module.dataset_stats.image_size.height: 32
        dataset.data_module.dataset_stats.image_size.width: 32
        dataset.data_module.dataset_stats.mean: src.dataset.base.RgbColor
        dataset.data_module.dataset_stats.mean.r: 0.49139968
        dataset.data_module.dataset_stats.mean.g: 0.48215841
        dataset.data_module.dataset_stats.mean.b: 0.44653091
        dataset.data_module.dataset_stats.std: src.dataset.base.RgbColor
        dataset.data_module.dataset_stats.std.r: 0.24703223
        dataset.data_module.dataset_stats.std.g: 0.24348513
        dataset.data_module.dataset_stats.std.b: 0.26158784
        dataset.data_module.data_root_dir: ./data
        dataset.data_module.batch_size: 128
        dataset.data_module.num_workers: 2
        optimizer: torch.optim.SGD
        optimizer.lr: 0.1
        optimizer.momentum: 0.9
        optimizer.weight_decay: 0.0005
        optimizer.dampening: 0.0005
        optimizer.nesterov: False

    """
    flattened_dict = {}
    for key, value in target_dict.items():
        if isinstance(value, MutableMapping):
            flattened_dict.update(
                {f"{key}{separator}{k}": v for k, v in flatten_dict(value).items()}
            )
        elif isinstance(value, MutableSequence):
            raise NotImplementedError("MutableSequence is not supported now.")
        else:
            flattened_dict[key] = value

    return flattened_dict


def unflatten_dict(
    flatten_dict: MutableMapping,
    separator: str = ".",
) -> Dict:
    """平坦化されたマッピング型の変数を受け取り, 階層化して返す.

    Args:
        flatten_dict (MutableMapping): 平坦化されたマッピング型の変数
        separator (str): 平坦化された辞書のキーの区切り文字

    Returns:
        Dict: 階層化した辞書

    """
    unflattened_dict = {}
    for key, value in flatten_dict.items():
        keys = key.split(separator)
        d = unflattened_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        last_key = keys[-1]
        d[last_key] = value

    return unflattened_dict


if __name__ == "__main__":
    config = {
        "classifier": {
            "encoder_base": {
                "_target_": "timm.create_model",
                "model_name": "efficientnet_b4",
                "pretrained": False,
                "num_classes": 1,
            },
        },
        "dataset": {
            "data_module": {
                "_target_": "src.dataset.cifar10.Cifar10DataModule",
                "dataset_stats": {
                    "num_classes": 10,
                    "image_size": {
                        "_target_": "src.dataset.base.ImageSize",
                        "height": 32,
                        "width": 32,
                    },
                    "mean": {
                        "_target_": "src.dataset.base.RgbColor",
                        "r": 0.49139968,
                        "g": 0.48215841,
                        "b": 0.44653091,
                    },
                    "std": {
                        "_target_": "src.dataset.base.RgbColor",
                        "r": 0.24703223,
                        "g": 0.24348513,
                        "b": 0.26158784,
                    },
                },
                "data_root_dir": "./data",
                "batch_size": 128,
                "num_workers": 2,
            },
        },
        "optimizer": {
            "_target_": "torch.optim.SGD",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "dampening": 0.0005,
            "nesterov": False,
        },
    }

    assert config == unflatten_dict(flatten_dict(config))
