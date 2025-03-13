from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"7_all_bands"
    tasks = {
        "algorithms": ["cnn_4200"],
        "datasets": ["lucas"],
        "target_sizes": [8, 16, 32, 64, 128, 256, 512],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
