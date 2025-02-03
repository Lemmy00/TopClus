export CUDA_VISIBLE_DEVICES=0

python src/trainer.py --model bert-base-multilingual-uncased --dataset yelp --n_clusters 100 --lr 5e-4 --cluster_weight 0.1 --seed 42 --do_cluster --do_inference
