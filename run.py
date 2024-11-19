from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p200"
    tasks = {
        "algorithms" : ["bsdrcnn"],
        "datasets": ["lucas_r"],
        "target_sizes" : [200]
    }
    ev = TaskRunner(tasks,tag,verbose=True)
    ev.evaluate()