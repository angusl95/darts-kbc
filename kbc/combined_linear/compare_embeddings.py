import os

for op in ['identity', 'conv', 'dil_conv', 'max_pool', 'avg_pool']:
			contents = """#$ -l gpu=1
#$ -l tmem=8G
#$ -l h_rt=2:00:00
#$ -S /bin/bash
#$ -N train
#$ -cwd

hostname

python /home/angulamb/darts-kbc/kbc/interleaved/train_from_embeddings.py --embeddings {} --dataset WN18RR --learning_rate 1e-4 \
--learning_rate_min 1e-4 --emb_dim 200 --channels 32 --arch {} --epochs 200 --batch_size 256 \
--reg 0 --report_freq 3 --optimizer Adam --layers 1 --weight_decay 1e-3 --seed 123 \
			""".format(op, 'embeddings_' + op + '.pt')
			with open("autosub.sub", "w") as file:
				file.write(contents)
			os.system("qsub autosub.sub")