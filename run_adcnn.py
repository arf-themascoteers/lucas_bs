from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = True
    tag = f"3_compo"
    tasks = {
        "algorithms": ["bsdrfc_r_norder","bsdrcnn_r"],
        "datasets": ["lucas"],
        "target_sizes" : [16],
        "scale_y": ["robust"],
        "mode": ["dyn"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
