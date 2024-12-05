from task_runner import TaskRunner

if __name__ == '__main__':
    TEST = False
    tag = "check"
    tasks = {
        "algorithms" : ["bsdrcnn","bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [8,16,32,65,131,262,525,1050,2100,4200],
        "scale_y" : ["robust"],
        "mode" : ["static","dyn","semi"]
    }
    folds = 10
    verbose = False

    if TEST:
        tasks = {
            "algorithms": ["bsdrcnn"],
            "datasets": ["min_lucas"],
            "target_sizes": [8],
            "scale_y": ["robust"],
            "mode": ["static"]
        }
        folds = 1
        verbose = True

    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()