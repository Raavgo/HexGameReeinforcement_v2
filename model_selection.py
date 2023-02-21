import os
import shutil
from  Utility import getArgs, get_epoch

def select_model():
    arg = getArgs()
    path = './numpy_bin'
    threshold = 0.55

    eval_path = arg["eval_path"]
    model_path = arg["model_path"]

    n = len(os.listdir(eval_path))
    if n == 0:
        exit()
    s = .0
    w = 0

    for file in os.listdir(eval_path):
        if os.path.isdir(f"{eval_path}/{file}"):
            n-=1
            continue
        with open(f"{eval_path}/{file}") as f:
            s += float(f.readline())
            w += int(f.readline())
        os.remove(f"{eval_path}/{file}")

    if s/n >= threshold:
        print('Better Model found')
        current_model = model_path + '/current_model.ckpt'
        best_model = model_path + "/best_model.ckpt"

        if os.path.isfile(best_model):
            os.remove(best_model)
        shutil.copy(current_model, best_model)
    else:
        print('No Model found')

    with open('logs/main_log.out', "a") as f:
        f.write(f"EPOCH: {get_epoch(path)-1}: Current Model won {w} games and has a WR of {s/n:.2f}\n\n")





if __name__ == "__main__":
    select_model()