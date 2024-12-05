from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "fc"
    tasks = {
        "algorithms" : ["bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [16,50,500],
        "scale_y" : ["robust","minmax"],
        "mode" : ["static"]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()