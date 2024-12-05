from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "check"
    tasks = {
        "algorithms" : ["bsdrcnn","bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [16,50, 100,200,300,500,1000,2000, 4200],
        "scale_y" : ["robust"],
        "mode" : ["static","dyn","semi"]
    }
    ev = TaskRunner(tasks,10,tag,verbose=False)
    ev.evaluate()