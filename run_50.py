from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "50"
    tasks = {
        "algorithms" : ["bsdrcnn_50_dyn","bsdrcnn_50_st","bsdrcnn_50_semi"],
        "datasets": ["lucas"],
        "target_sizes" : [50]
    }
    ev = TaskRunner(tasks,10,tag,verbose=False)
    ev.evaluate()