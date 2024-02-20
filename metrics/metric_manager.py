import os
import numpy as np
from typing import Optional, Callable, Dict, List, Any

from cl_gym.utils.callbacks.helpers import IntervalCalculator, Visualizer
from cl_gym.utils.metrics import ContinualMetric, PerformanceMetric, ForgettingMetric

from cl_gym.utils.callbacks.metric_manager import MetricCollector

class GeneralMetric():
    def __init__(self, num_tasks: int, epochs_per_task: Optional[int] = 1, agg = np.mean):
        self.num_tasks = num_tasks
        self.epochs_per_task = epochs_per_task
        # data shape => [(task_learned+1) x (task_evaluated+1) x epoch_per task]
        # The 0 index is reserved for 'initialization' metrics
        self.agg = agg
        self.data = np.zeros((num_tasks+1, num_tasks+1, epochs_per_task)).tolist()
        self.dtype = None 

    def update(self, task_learned: int, task_evaluated: int, value: Any, epoch: Optional[int] = 1):
        if epoch < 1:
            raise ValueError("Epoch number is 1-based")
        if self.dtype is None:
            self.dtype = type(value)
        self.data[task_learned][task_evaluated][epoch-1] = value
    
    def compute(self, current_task: int) -> float:
        if current_task < 1:
            raise ValueError("Tasks are 1-based. i.e., the first task's id is 1, not 0.")
        if self.dtype is None:
            raise NotImplementedError
        else:
            total = self.dtype()
        for diff in [self.data[current_task][i][-1] for i in range(1,current_task+1)]:
            if self.dtype == list:
                total+=diff
            elif self.dtype == dict:
                total.update(diff)
            else:
                raise NotImplementedError
        return self.agg(total)
        
    def compute_final(self) -> float:
        return self.compute(self.num_tasks)

    def compute_overall(self) -> float:
        overall = list()
        for t in range(1, self.num_tasks+1):
            overall.append(self.compute(t))
        return overall

    def get_data(self, r = 3) -> list:
        overall = list()
        agg = self.agg
        for t in range(1, self.num_tasks+1):
            self.agg = lambda x:x
            overall.append(self.compute(t))
        self.agg = agg
        return overall


class ClasswiseAccuracy(GeneralMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg = lambda x:x 


class PerformanceMetric2(PerformanceMetric):
    def compute(self, current_task: int) -> float:
        if current_task < 1:
            raise ValueError("Tasks are 1-based. i.e., the first task's id is 1, not 0.")
        return float(np.mean(self.data[current_task, 1:current_task+1, -1]))
        
    def compute_final(self) -> float:
        return self.compute(self.num_tasks)
    
    def get_data(self, r = 3) -> np.ndarray:
        return np.round(self.data[1:,1:,-1], r)
    
    def compute_overall(self) -> float:
        data = self.data[1:,1:,-1]
        return [np.mean(i[np.nonzero(i)]) for i in data]


class StdMetric(PerformanceMetric2):
    def __init__(self, 
                 num_tasks: int, 
                 acc_metric: PerformanceMetric = None,
                 epochs_per_task: Optional[int] = 1,
                 per_task_classes: Optional[int] = 2,
                 ):
        self.num_tasks = num_tasks
        self.acc_metric = acc_metric
        self.epochs_per_task = epochs_per_task
        self.per_task_classes = per_task_classes
        # data shape => [(task_learned+1) x (task_evaluated+1) x epoch_per task]
        # The 0 index is reserved for 'initialization' metrics
        self.data = np.zeros((num_tasks+1, num_tasks+1, epochs_per_task))

    def update_acc_metric(self, acc_metric: PerformanceMetric):
        self.acc_metric = acc_metric

    def get_std(self) -> list:
        std = list()
        std_data = self.data[1:,1:,-1]
        mean_data = self.acc_metric.data[1:,1:,-1]
        for i, e in enumerate(mean_data):
            task_mean = np.mean(e[:i+1])
            square_sum = 0
            for j in range(i+1):
                square_sum += self.per_task_classes*((task_mean - e[j])**2 + std_data[i,j]**2)
            task_std = np.sqrt(square_sum / ((i+1)*self.per_task_classes))
            # print(task_mean)
            std.append(task_std)
        return std


class MetricCollector2(MetricCollector):
    """
    Collects metrics during the learning.
    This callback can support various metrics such as average accuracy/error, and average forgetting.
    """
    def __init__(self, num_tasks: int,
                 epochs_per_task: Optional[int] = 1,
                 collect_on_init: bool = False,
                 collect_metrics_for_future_tasks: bool = False,
                 eval_interval: str = 'epoch',
                 eval_type: str = 'classification',
                 tuner_callback: Optional[Callable[[float, bool], None]] = None):
        """
        Args:
            num_tasks: The number of task for the learning experience.
            epochs_per_task: The number of epochs per task.
            collect_on_init: Should also collect metrics before training starts?
            collect_metrics_for_future_tasks: Should collect metrics for future tasks? (e.g.,, for forward-transfer)
            eval_interval: The intervals at which the algorithm will be evaluated. Can be either `task` or `epoch`
            eval_type: Is this a `classification` task or `regression` task?
            tuner_callback: Optional tuner callback than can be called with eval metrics for parameter optimization.
        """
        super().__init__(num_tasks, epochs_per_task = epochs_per_task, collect_on_init = collect_on_init,
                        collect_metrics_for_future_tasks = collect_metrics_for_future_tasks,
                         eval_interval = eval_interval, eval_type = eval_type, tuner_callback = tuner_callback)

    def _prepare_meters(self) -> Dict[str, ContinualMetric]:
        if self.eval_type == 'classification':
            metrics = {'accuracy': PerformanceMetric2(self.num_tasks, self.epochs_per_task),
                    'std': StdMetric(self.num_tasks, self.epochs_per_task),
                    'forgetting': ForgettingMetric(self.num_tasks, self.epochs_per_task),
                    'loss': PerformanceMetric(self.num_tasks, self.epochs_per_task)}
            metrics['std'].update_acc_metric(metrics['accuracy'])
            return metrics
        else:
            return {'loss': PerformanceMetric(self.num_tasks, self.epochs_per_task)}

    def _update_meters(self, task_learned: int, task_evaluated: int, metrics: dict, relative_step: int):
        if self.eval_type == 'classification':
            self.meters['loss'].update(task_learned, task_evaluated, metrics['loss'], relative_step)
            self.meters['accuracy'].update(task_learned, task_evaluated, metrics['accuracy'], relative_step)
            self.meters['std'].update(task_learned, task_evaluated, metrics['std'], relative_step)
            self.meters['forgetting'].update(task_learned, task_evaluated, metrics['accuracy'], relative_step)
        else:
            self.meters['loss'].update(task_learned, task_evaluated, metrics['loss'], relative_step)

    def _update_logger(self, trainer, task_evaluated: int, metrics: dict, global_step: int):
        if trainer.logger is None:
            return
        
        if self.eval_type == 'classification':
            trainer.logger.log_metric(f'acc_{task_evaluated}', round(metrics['accuracy'], 2), global_step)
            trainer.logger.log_metric(f'std_{task_evaluated}', round(metrics['std'], 2), global_step)
            trainer.logger.log_metric(f'loss_{task_evaluated}', round(metrics['loss'], 2), global_step)
            if trainer.current_task > 0:
                avg_acc = round(self.meters['accuracy'].compute(trainer.current_task), 2)
                trainer.logger.log_metric(f'average_acc', avg_acc, global_step)
        else:
            trainer.logger.log_metric(f'loss_{task_evaluated}', round(metrics['loss'], 2), global_step)
            if trainer.current_task > 0:
                avg_loss = round(self.meters['loss'].compute(trainer.current_task), 5)
                trainer.logger.log_metric(f'average_loss', avg_loss, global_step)

    def _update_tuner(self, is_final_score: bool):
        if self.tuner_callback is None:
            return
        if self.eval_type == 'classification':
            score = self.meters['accuracy'].compute_final()
        else:
            score = self.meters['loss'].compute_final()
        self.tuner_callback(score, is_final_score)

    def log_metrics(self, trainer, task_learned: int, task_evaluated: int,
                     metrics: dict, global_step: int, relative_step: int):
        self._update_meters(task_learned, task_evaluated, metrics, relative_step)
        self._update_logger(trainer, task_evaluated, metrics, global_step)
        self._update_tuner(is_final_score=False)
    
    def _collect_eval_metrics(self, trainer, start_task: int, end_task: int):
        global_step = trainer.current_task if self.eval_interval == 'task' else trainer.current_epoch
        relative_step = self.interval_calculator.get_step_within_task(trainer.current_epoch)
        for eval_task in range(start_task, end_task + 1):
            task_metrics = trainer.validate_algorithm_on_task(eval_task)
            self.log_metrics(trainer, trainer.current_task, eval_task, task_metrics, global_step, relative_step)
            print(f"[{global_step}] Eval metrics for task {eval_task} >> {task_metrics}")

    def save_metrics(self):
        metrics = ['accuracy', 'loss'] if self.eval_type == 'classification' else ['loss']
        for metric in metrics:
            filepath = os.path.join(self.save_paths['metrics'], metric + ".npy")
            with open(filepath, 'wb') as f:
                np.save(f, self.meters[metric].data)


