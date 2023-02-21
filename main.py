import subprocess
import time

result = subprocess.run(['squeue'], stdout=subprocess.PIPE)
n = 0
commands = {
    1: ["sbatch","1_generator_slurm.sh"],
    2: ["sbatch","2_concate_slurm.sh"],
    3: ["sbatch","3_train_slurm.sh"],
    4: ["sbatch","4_evaluate_slurm.sh"],
    5: ["sbatch","5_modelselect_slurm.sh"],
}
counter = 1
free = True
while n < 50:
    if free:
        subprocess.run(commands[counter])
        free = False

    time.sleep(10)
    result = subprocess.run(['squeue'], stdout=subprocess.PIPE)
    result = str(result).replace("main_slu ai21m034", "")
    if "ai21m034" not in str(result):
        free = True
        counter += 1
        if counter > 5:
            n += 1
            counter = 1





