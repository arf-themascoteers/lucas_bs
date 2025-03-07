from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.95]
    verbose = False
    tag = f"12_best_95_32_64"
    tasks = {
        "algorithms": ["bsdrcnn_r"],
        "datasets": ["lucas"],
        "target_sizes": [32,64,256],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
