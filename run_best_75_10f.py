from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"20_best_75_10f"
    tasks = {
        "algorithms": ["bsdrcnn_r"],
        "datasets": ["lucas"],
        "target_sizes": [8, 16, 128],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 10
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
