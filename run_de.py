from task_runner import TaskRunner
from my_env import TEST

if __name__ == '__main__':
    train_sizes = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    if TEST:
        train_sizes = [0.05, 0.15]
    tag = f"de6"
    tasks = {
        "algorithms" : ["bsdrcnn"],
        "datasets": ["lucas"],
        "target_sizes" : [8,16,32,64,128,256,512],
        "scale_y" : ["robust"],
        "mode" : ["dyn"],
        "train_sizes":train_sizes
    }
    if TEST:
        tasks = {
            "algorithms": ["bsdrcnn"],
            "datasets": ["min_lucas"],
            "target_sizes": [8, 16, 32],
            "scale_y": ["robust"],
            "mode": ["dyn"],
            "train_sizes": train_sizes
        }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=True)
    ev.evaluate()
