from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "100"
    tasks = {
        "algorithms" : ["bsdrcnn_100_dyn","bsdrcnn_100_st","bsdrcnn_100_semi"],
        "datasets": ["lucas"],
        "target_sizes" : [100]
    }
    ev = TaskRunner(tasks,1,tag,verbose=False)
    ev.evaluate()