import subprocess

seeds = [1,2,3,4,5]

for seed in seeds:

    #pre-train

    command = f'python main_sequence_lm.py --lang python --n_epochs 5 --data code --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang python --n_epochs 5 --data desc --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 6L-python --n_epochs 5 --data code --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 6L-python --n_epochs 5 --data desc --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 5L-python --n_epochs 5 --data code --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 5L-python --n_epochs 5 --data desc --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    #random init

    command = f'python main_code_retrieval.py --lang python --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 6L-python --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 5L-python --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang python --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 6L-python --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 5L-python --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    #fine tune

    command = f'python main_code_retrieval.py --lang python --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 6L-python --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 5L-python --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang python --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 6L-python --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 5L-python --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()