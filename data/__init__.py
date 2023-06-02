from .cf_loader import Task1Loader, Task2Loader

SUPPORTED_DATASETS = {
    "task1": Task1Loader,
    "task2": Task2Loader
}

def get_dataset(args):
    dataset_name = args.dataset
    if dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError("Dataset {} does not exist.".format(dataset_name))
    if (dataset_name == 'task1') or (dataset_name == 'task2'):
        return SUPPORTED_DATASETS[dataset_name](args.data_path)