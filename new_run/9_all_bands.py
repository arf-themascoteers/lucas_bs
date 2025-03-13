from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    train_sizes = [0.05, 0.15, 0.25, 0.75]
    verbose = False
    tag = f"9_all_bands_epoch"
    tasks = {
        "algorithms": ["bsdrcnn_r_4200_3"],
        "datasets": ["lucas"],
        "target_sizes": [4200],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 5
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
