from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "asa"
    tasks = {
        "algorithms" : ["asa"],
        "datasets": ["lucas"],
        "target_sizes" : [500]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()