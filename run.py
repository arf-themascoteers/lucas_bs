from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "baseline"
    tasks = {
        "algorithms" : ["bsdrcnn"],
        "datasets": ["lucas_asa"],
        "target_sizes" : [200]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()