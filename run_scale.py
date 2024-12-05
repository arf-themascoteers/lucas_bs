from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "fc"
    tasks = {
        "algorithms" : ["bsdrfc"],
        "datasets": ["lucas"],
        "target_sizes" : [16,50,100,200,300,500,1000,2000,4200],
        "scale_y" : ["robust","minmax"],
        "mode" : ["static","dynamic","semi"]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()