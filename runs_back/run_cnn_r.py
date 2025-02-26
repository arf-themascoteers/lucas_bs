from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.05]
    verbose = False
    tag = f"rpd"
    tasks = {
        "algorithms": ["bsdrcnn_r"],
        "datasets": ["min_lucas"],
        "target_sizes": [8],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 5
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
