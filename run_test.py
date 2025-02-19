from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.05, 0.15]
    verbose = False
    tag = f"test1"
    tasks = {
        "algorithms": ["bsdrfc"],
        "datasets": ["min_lucas"],
        "target_sizes": [8, 16, 32, 64, 128, 256, 512],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 10
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
