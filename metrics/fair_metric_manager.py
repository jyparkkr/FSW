import os
import numpy as np
from typing import Optional, Callable, Dict

from cl_gym.utils.callbacks.helpers import IntervalCalculator, Visualizer
from cl_gym.utils.metrics import ContinualMetric, PerformanceMetric, ForgettingMetric

from cl_gym.utils.callbacks.metric_manager import MetricCollector
from .metric_manager import PerformanceMetric2, GeneralMetric, ClasswiseAccuracy


class FairnessMetric(GeneralMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg = kwargs.get('agg', np.max)
    
class FairMetricCollector(MetricCollector):
    """
    Collects metrics during the learning.
    This callback can support various metrics such as average accuracy/error, and average forgetting.
    """
    fairness_metric = "EO"
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
        if FairMetricCollector.fairness_metric.lower() == "eo":
            self.fairness_metric = FairnessMetric
        elif FairMetricCollector.fairness_metric.lower() == "dp":
            self.fairness_metric = FairnessMetric
        else:
            raise NotImplementedError

        super().__init__(num_tasks, epochs_per_task = epochs_per_task, collect_on_init = collect_on_init,
                        collect_metrics_for_future_tasks = collect_metrics_for_future_tasks,
                         eval_interval = eval_interval, eval_type = eval_type, tuner_callback = tuner_callback)

    def _prepare_meters(self) -> Dict[str, ContinualMetric]:
        if self.eval_type == 'classification':
            return {'accuracy': PerformanceMetric2(self.num_tasks, self.epochs_per_task),
                    'accuracy_s0': PerformanceMetric2(self.num_tasks, self.epochs_per_task),
                    'accuracy_s1': PerformanceMetric2(self.num_tasks, self.epochs_per_task),
                    'classwise_accuracy': ClasswiseAccuracy(self.num_tasks, self.epochs_per_task),
                    'fairness': FairnessMetric(self.num_tasks, self.epochs_per_task),
                    'forgetting': ForgettingMetric(self.num_tasks, self.epochs_per_task),
                    'loss': PerformanceMetric(self.num_tasks, self.epochs_per_task)}
        else:
            return {'loss': PerformanceMetric(self.num_tasks, self.epochs_per_task)}

    def _update_meters(self, task_learned: int, task_evaluated: int, metrics: dict, relative_step: int):
        if self.eval_type == 'classification':
            self.meters['loss'].update(task_learned, task_evaluated, metrics['loss'], relative_step)
            self.meters['accuracy'].update(task_learned, task_evaluated, metrics['accuracy'], relative_step)
            self.meters['accuracy_s0'].update(task_learned, task_evaluated, metrics['accuracy_s0'], relative_step)
            self.meters['accuracy_s1'].update(task_learned, task_evaluated, metrics['accuracy_s1'], relative_step)
            self.meters['classwise_accuracy'].update(task_learned, task_evaluated, metrics['classwise_accuracy'], relative_step)
            self.meters['fairness'].update(task_learned, task_evaluated, metrics['fairness'], relative_step)
            self.meters['forgetting'].update(task_learned, task_evaluated, metrics['accuracy'], relative_step)
        else:
            self.meters['loss'].update(task_learned, task_evaluated, metrics['loss'], relative_step)

    def _update_logger(self, trainer, task_evaluated: int, metrics: dict, global_step: int):
        if trainer.logger is None:
            return
        
        if self.eval_type == 'classification':
            trainer.logger.log_metric(f'acc_{task_evaluated}', round(metrics['accuracy'], 2), global_step)
            trainer.logger.log_metric(f'acc_s0_{task_evaluated}', round(metrics['accuracy_s0'], 2), global_step)
            trainer.logger.log_metric(f'acc_s1_{task_evaluated}', round(metrics['accuracy_s1'], 2), global_step)
            # trainer.logger.log_metric(f'multi_eo_{task_evaluated}', round(metrics['fairness'], 2), global_step)
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

    def save_metrics(self):
        metrics = ['accuracy', 'loss'] if self.eval_type == 'classification' else ['loss']
        for metric in metrics:
            filepath = os.path.join(self.save_paths['metrics'], metric + ".npy")
            with open(filepath, 'wb') as f:
                np.save(f, self.meters[metric].data)


