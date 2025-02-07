from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "16"
    tasks = {
        "algorithms" : ["bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [16],
        "scale_y" : ["robust"],
        "mode" : ["dyn"]
    }
    folds = 1

    ev = TaskRunner(tasks,folds,tag,verbose=True)
    ev.evaluate()
