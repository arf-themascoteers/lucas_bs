from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"rr"
    tasks = {
        "algorithms": ["bsdrcnn_r","bsdrcnn","bsdrfc_r","bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes": [128],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 5
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
