# Support Ticket Classification

Run training script

```bash
python main.py --train_data_path ../data/voda_idea_data_splits/train.csv --val_data_path ../data/voda_idea_data_splits/val.csv --model_name bert_base_uncased.ckpt --hf_lm bert-base-uncased --n_epochs 10 --train_batch_size 8 --test_batch_size 16 --grad_accum_steps 1 --num_workers 4 --accelerator gpu --precision_strategy 16-mixed --n_devices 2 --gpu_strategy ddp --use_wandb 1 --wandb_name bert-base

```
