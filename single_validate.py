import torch
from torch.utils import model_zoo
from torch.utils.data import DataLoader, SequentialSampler

from datasets import build_dataset
from models.convnextv2 import convnextv2_pico

from engine_finetune import evaluate

# Dataset
args = object()
args.data_set = 'IMNET'
args.data_path = '' # path to ImgetNet


dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
dataset_val, _ = build_dataset(is_train=False, args=args)
sampler_val = SequentialSampler(dataset_val)
data_loader_val = DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=4,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)

# Model
device = torch.device('cuda:0')
model = convnextv2_pico()
checkpoint = model_zoo.load_url('', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.to(device)

# Evaluate
test_res = evaluate(data_loader_val, model, 'cuda:0')
print(f"Accuracy of the network on {len(dataset_val)} test images: {test_res['acc1']:.5f}%")

