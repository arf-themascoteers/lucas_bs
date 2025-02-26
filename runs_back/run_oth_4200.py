from task_runner import TaskRunner

if __name__ == '__main__':
    train_sizes = [0.75]
    verbose = False
    tag = f"cnn_static_4200"
    tasks = {
        "algorithms": ["bsdrcnn_r_4200"],
        "datasets": ["lucas"],
        "target_sizes" : [4200],
        "scale_y": ["robust"],
        "mode": ["static"],
        "train_sizes": train_sizes
    }
    folds = 1
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
