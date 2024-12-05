from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "asa"
    tasks = {
        "algorithms" : ["asa"],
        "datasets": ["lucas"],
        "target_sizes" : [500],
        "scale_y" : ["robust"],
        "mode" : ["static"]
    }
    folds = 10
    verbose = True
    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()