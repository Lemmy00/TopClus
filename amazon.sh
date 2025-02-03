export CUDA_VISIBLE_DEVICES=0

python src/trainer.py --model bert-base-multilingual-uncased --dataset amazon_sampled --n_clusters 100 --lr 5e-4 --batch_size 64 --k 5 --cluster_weight 0.1 --seed 42 --pretrain_epoch 30 --cluster_weight 30 --do_cluster --do_inference
