from task_runner import TaskRunner

if __name__ == '__main__':
    TEST = False
    tag = "check"
    tasks = {
        "algorithms" : ["bsdrcnn","bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [16,50, 100,200,300,500,1000,2000, 4200],
        "scale_y" : ["robust"],
        "mode" : ["static","dyn","semi"]
    }
    folds = 10
    verbose = False

    if TEST:
        tasks = {
            "algorithms": ["bsdrcnn"],
            "datasets": ["lucas"],
            "target_sizes": [16],
            "scale_y": ["robust"],
            "mode": ["static"]
        }
        folds = 1
        verbose = True

    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()