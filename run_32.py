from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "32"
    tasks = {
        "algorithms" : ["bsdrfc_32_dyn","bsdrkan_32_dyn"],
        "datasets": ["lucas"],
        "target_sizes" : [32]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()