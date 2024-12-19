from task_runner import TaskRunner

if __name__ == '__main__':
    TEST = False
    tag = "all_bands2fc"
    tasks = {
        "algorithms" : ["bsdrfc_static"],
        "datasets": ["lucas"],
        "target_sizes" : [4200],
        "scale_y" : ["robust"],
        "mode" : ["static"]
    }
    folds = 1
    verbose = False


    ev = TaskRunner(tasks,folds,tag,verbose=verbose)
    ev.evaluate()
