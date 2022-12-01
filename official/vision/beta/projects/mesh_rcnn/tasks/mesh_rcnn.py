"""MeshRCNN task definition."""

from official.core import base_task
from official.core import task_factory
from official.vision.beta.projects.mesh_rcnn.configs import mesh_rcnn as exp_cfg
from official.vision.beta.projects.mesh_rcnn.modeling import factory
from official.modeling.optimization.optimizer_factory import OptimizerFactory
from official.modeling.optimization import ema_optimizer
from official.modeling import performance

import tensorflow as tf

@task_factory.register_task_cls(exp_cfg.MeshRCNNTask)
class MeshRCNNTask(base_task.Task):
  """A single-replica view of training procedure.

    MeshRCNN task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """
    def __init__(self, params, logging_dir: Optional[str] = None):
        super().__init__(params, logging_dir)
        return

    def build_model(self):
        """Build Mesh R-CNN model."""

        input_specs = tf.keras.layers.InputSpec(
            shape=[None] + self.task_config.model.input_size)

        l2_weight_decay = self.task_config.losses.l2_weight_decay # either T or F
        
        # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
        # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
        # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
        l2_regularizer = (tf.keras.regularizers.l2(
            l2_weight_decay / 2.0) if l2_weight_decay else None)

        model, losses = factory.build_mesh_rcnn(
            input_specs, model_base_cfg, l2_regularizer)

        if self.task_config.freeze_backbone:
            model.backbone.trainable = False

        return model

    def build_inputs(self, params, input_context=None):
        """Build input dataset."""
        return
    
    def build_metrics(self, training=True):
        """Build metrics."""
        return

    def build_losses(self, outputs, labels, aux_losses=None):
        """Build Mesh RCNN losses."""
        return

    def initialize(self, model: tf.keras.Model):
        """Loading pretrained checkpoint."""
        return

    def train_step(self, inputs, model, optimizer, metrics=None):
        """Train Step.

        Forward step and backwards propagate the model.

        Args:
        inputs: a dictionary of input tensors.
        model: the model, forward pass definition.
        optimizer: the optimizer for this training step.
        metrics: a nested structure of metrics objects.

        Returns:
        A dictionary of logs.
        """
        return

    def validation_step(self, inputs, model, metrics=None):
        """Validatation step.

        Args:
        inputs: a dictionary of input tensors.
        model: the keras.Model.
        metrics: a nested structure of metrics objects.

        Returns:
        A dictionary of logs.
        """
        return
    
    def aggregate_logs(self, state=None, step_outputs=None):
        """Get Metric Results."""
        return

    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Reduce logs and remove unneeded items. Update with results."""
        return

    def create_optimizer(self,
                       optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
        """Creates an TF optimizer from configurations.

        Args:
        optimizer_config: the parameters of the Optimization settings.
        runtime_config: the parameters of the runtime.

        Returns:
        A tf.optimizers.Optimizer object.
        """
        opt_factory = optimizer_factory.OptimizerFactory(optimizer_config)
        # pylint: disable=protected-access
        ema = opt_factory._use_ema
        opt_factory._use_ema = False

        opt_type = opt_factory._optimizer_type
        '''
        if opt_type == 'sgd_torch':
            optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
            optimizer.set_bias_lr(
                opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))
            optimizer.search_and_set_variable_groups(self._model.trainable_variables)
        '''
        optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
        opt_factory._use_ema = ema

        if ema:
            logging.info('EMA is enabled.')
            optimizer = ema_optimizer.ExponentialMovingAverage(optimizer, **self._ema_config.as_dict())
            #optimizer = opt_factory.add_ema(optimizer)

        # pylint: enable=protected-access

        if runtime_config and runtime_config.loss_scale:
            use_float16 = runtime_config.mixed_precision_dtype == 'float16'
            optimizer = performance.configure_optimizer(
                optimizer,
                use_float16=use_float16,
                loss_scale=runtime_config.loss_scale)

        return optimizer
      
