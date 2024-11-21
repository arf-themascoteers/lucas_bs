from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "200"
    tasks = {
        "algorithms" : ["bsdrcnn_200_dyn","bsdrcnn_200_st","bsdrcnn_200_semi"],
        "datasets": ["lucas"],
        "target_sizes" : [200]
    }
    ev = TaskRunner(tasks,10,tag,verbose=False)
    ev.evaluate()