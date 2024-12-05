from task_runner import TaskRunner

if __name__ == '__main__':
    TEST = False
    tag = "check"
    tasks = {
        "algorithms" : ["bsdrcnn_r","bsdrfc_r"],
        "datasets": ["lucas"],
        "target_sizes" : [8,16,32,64,128,256,512,1024,2048,500,4200],
        "scale_y" : ["robust"],
        "mode" : ["dyn","semi"]
    }
    folds = 10
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
