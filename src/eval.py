import _init_paths
from opts import opts
from dataset.dataset_factory import dataset_factory

if __name__ == '__main__':
    opt = opts().parse()
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    dataset.run_eval_only(opt.save_dir)