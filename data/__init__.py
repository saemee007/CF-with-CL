from .cifar import CIFAR10Contrastive
# from .cifar import CIFAR, CIFARContrastive
# from .imagenet import Imagenet, ImagenetContrastive

SUPPORTED_DATASETS = {
    # "cifar10": CIFAR,
    "cifar10_cl": CIFAR10Contrastive,
    # "imagenet": Imagenet,
    # "imagenet_cl": ImagenetContrastive
}

def get_dataset(args, aug, return_label=False):
    dataset_name = args.dataset
    noise_type = args.noise_type
    openset_ratio = args.openset_ratio
    closeset_ratio = args.closeset_ratio
    if dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError("Dataset {} does not exist.".format(dataset_name))
    return SUPPORTED_DATASETS[dataset_name](aug=aug,noise_type=noise_type, openset_ratio=openset_ratio, closeset_ratio=closeset_ratio, return_label=return_label)