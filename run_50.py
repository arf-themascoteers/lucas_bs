from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "50_2"
    tasks = {
        "algorithms" : ["bsdrcnn_50_dyn"],
        "datasets": ["lucas"],
        "target_sizes" : [50]
    }
    ev = TaskRunner(tasks,1,tag,verbose=False)
    ev.evaluate()