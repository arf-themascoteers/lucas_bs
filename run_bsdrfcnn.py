from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "bsdrcnn_robust_crop"
    tasks = {
        "algorithms" : ["bsdrcnn"],
        "datasets": ["lucas_crop_asa"],
        #"target_sizes" : [20,50,100,200,300,500,600]
        "target_sizes" : [200]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()