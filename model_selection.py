import os
import shutil
from  Utility import getArgs

def select_model():
    arg = getArgs()
    threshold = 0.6

    eval_path = arg["eval_path"]
    model_path = arg["model_path"]

    n = len(os.listdir(eval_path))
    if n == 0:
        exit()
    sum = .0


    for file in os.listdir(eval_path):
        with open(f"{eval_path}/{file}") as f:
            sum += float(f.readline())
        os.remove(f"{eval_path}/{file}")

    if sum >= threshold:
        current_model = model_path + '/current_model.ckpt'
        best_model = model_path + "/best_model.ckpt"

        if os.path.isfile(model_path + '/best_model'):
            os.remove(model_path + '/best_model')
        shutil.copy(current_model, best_model)




if __name__ == "__main__":
    select_model()