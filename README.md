# MSc-Project
# Zizhuo Wang  University of Leeds  UID: 201406070

Data preprocess:
1. Python download_data.py
2. python bpe.py --lang python--vocab_max_size 10000 --bpe_pct 0.5
3. python create_vocab.py

Train:
1. python main_code_retrieval.py --lang python--model transformer --seed 1  
2. python main_code_retrieval.py --lang python--model lstm --seed 1

Prediction.py
1. python main_method_prediction.py --lang python --model transformer --seed 1
2. python main_method_prediction.py --lang python --model lstm --seed 1


Evaluation:
1. Python --lang python --model lstm --seed 2 --load
2. Python --lang python --model transformer --seed 2 --load
