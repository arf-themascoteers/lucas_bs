from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = True
    tag = f"16_best_75_16_1"
    tasks = {
        "algorithms": ["bsdrcnn_r"],
        "datasets": ["lucas"],
        "target_sizes": [16],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
