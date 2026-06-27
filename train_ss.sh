torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node=4 train_lightning.py \
--config ./configs/ss.yaml \
--vis ./train_sparse_strcuture
