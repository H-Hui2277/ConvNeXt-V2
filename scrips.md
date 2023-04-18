## ImageNet-1K Fine-Tuning

ConvNeXt V2-Base fine-tuning on ImageNet-1K with 4 8-GPU nodes:
```
python submitit_finetune.py --nodes 4 --ngpus 8 \
--model convnextv2_base \
--batch_size 32 \
--blr 6.25e-4 \
--epochs 100 \
--warmup_epochs 20 \
--layer_decay_type 'group' \
--layer_decay 0.6 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--data_path /path/to/imagenet-1k \
--job_dir /results
```

The following commands run the fine-tuning on a single machine:

```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_base \
--batch_size 32 --update_freq 4 \
--blr 6.25e-4 \
--epochs 100 \
--warmup_epochs 20 \
--layer_decay_type 'group' \
--layer_decay 0.6 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--data_path H:\\Datasets\\ImageNet\\imagenet \
--output_dir /results
```