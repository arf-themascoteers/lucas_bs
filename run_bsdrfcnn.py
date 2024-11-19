from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "bsdrcnn"
    tasks = {
        "algorithms" : ["bsdrcnn"],
        "datasets": ["lucas"],
        "target_sizes" : [500]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()