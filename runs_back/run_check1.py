from task_runner import TaskRunner

if __name__ == '__main__':
    TEST = False
    tag = "check1"
    tasks = {
        "algorithms" : ["bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [512],
        "scale_y" : ["robust"],
        "mode" : ["static","dyn","semi"]
    }
    folds = 1
    verbose = False

    if TEST:
        tasks = {
            "algorithms": ["bsdrcnn"],
            "datasets": ["min_lucas"],
            "target_sizes": [2048],
            "scale_y": ["robust"],
            "mode": ["static"]
        }
        folds = 1
        verbose = True

    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
