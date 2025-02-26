from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    verbose = True
    tag = f"cnnvsfc"
    tasks = {
        "algorithms": ["bsdrcnn_r","bsdrfc_r"],
        "datasets": ["lucas"],
        "target_sizes" : [128],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
