from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "fc16"
    tasks = {
        "algorithms" : ["bsdrfc_16_dyn"],
        "datasets": ["lucas"],
        "target_sizes" : [16]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()