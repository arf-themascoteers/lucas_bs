from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "bsdrfcnn"
    tasks = {
        "algorithms" : ["bsdrfcnn"],
        "datasets": ["min_lucas"],
        "target_sizes" : [20,50,100,200,300,500,600]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()