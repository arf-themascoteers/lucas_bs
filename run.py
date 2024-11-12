from task_runner import TaskRunner
import plotter_classification
import clean_up

if __name__ == '__main__':
    clean_up.do_it()
    tag = "p25"
    tasks = {
        "algorithms" : ["bsdrcnn"],
        "datasets": ["lucas_r"],
        "target_sizes" : [200]
    }
    ev = TaskRunner(tasks,tag,verbose=True)
    summary, details = ev.evaluate()
    plotter_classification.plot_combined(sources=["p25"],only_algorithms=["bsdrcnn"])