import torch
from torch.utils import model_zoo
from torch.utils.data import DataLoader, SequentialSampler

from datasets import build_dataset
from models.convnextv2 import convnextv2_pico, convnextv2_tiny

from engine_finetune import evaluate

model_urls = {
    'atto': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
    'femto': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
    'pico': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
    'nano': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt',
    'tiny': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt',
    'base': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt',
    'large': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt',
    'huge': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt',
}

# Dataset
class Args(object):
    def __init__(self) -> None:
        self.data_set = 'IMNET'
        self.data_path = 'H:\\Datasets\\ImageNet\\imagenet' # path to ImgetNet
        self.input_size = 224
        self.imagenet_default_mean_and_std = True
        self.crop_pct = None
args = Args()

if __name__ == '__main__':
    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # Model
    device = torch.device('cuda:0')
    model = convnextv2_tiny()
    checkpoint = model_zoo.load_url(model_urls['tiny'], map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Evaluate
    test_res = evaluate(data_loader_val, model, 'cuda:0')
    print(f"Accuracy of the network on {len(dataset_val)} test images: {test_res['acc1']:.5f}%")

