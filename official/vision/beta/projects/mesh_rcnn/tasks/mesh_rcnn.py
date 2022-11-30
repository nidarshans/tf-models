"""MeshRCNN task definition."""

from official.core import base_task
from official.core import task_factory
from official.vision.beta.projects.mesh_rcnn.configs import mesh_rcnn as exp_cfg
from official.vision.beta.projects.mesh_rcnn.modeling import factory

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
        #Test commit
        return





    




