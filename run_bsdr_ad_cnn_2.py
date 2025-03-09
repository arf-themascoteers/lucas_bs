from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"44_bsdr_ad_cnn_4"
    tasks = {
        "algorithms": ["bsdrcnn"],
        "datasets": ["lucas"],
        "target_sizes": [8, 16, 32, 64, 128],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
