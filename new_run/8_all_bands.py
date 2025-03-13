from task_runner import TaskRunner
import os

os.chdir("..")

if __name__ == '__main__':
    train_sizes = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    verbose = False
    tag = f"8_all_bands"
    tasks = {
        "algorithms": ["bsdrcnn_r_4200_2"],
        "datasets": ["lucas"],
        "target_sizes": [4200],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 5
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
