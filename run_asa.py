from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "asa_r"
    tasks = {
        "algorithms" : ["asa"],
        "datasets": ["lucas_crop"],
        "target_sizes" : [500],
        "scale_y" : ["robust"],
        "mode" : ["static"]
    }
    folds = 1
    verbose = True
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()