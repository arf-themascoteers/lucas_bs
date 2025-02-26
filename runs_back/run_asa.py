from task_runner import TaskRunner
from my_env import TEST

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"asa"
    tasks = {
        "algorithms" : ["asa"],
        "datasets": ["lucas_crop"],
        "target_sizes" : [500],
        "scale_y" : ["robust"],
        "mode" : ["dyn"],
        "train_sizes":train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
