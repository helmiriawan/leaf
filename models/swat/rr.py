import tensorflow as tf

from model import Model


number_features = 50


class ClientModel(Model):
    def __init__(self, lr, num_targets):
        self.num_targets = num_targets
        super(ClientModel, self).__init__(lr)

    def create_model(self):
        """Model function for Ridge Regression."""

        features = tf.placeholder(
            tf.float32, shape=[None, number_features], name='features')
        targets = tf.placeholder(tf.float32, shape=[None], name='targets')
        
        w = tf.get_variable(
            'weights', 
            shape=[number_features, self.num_targets],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer
        )
        b = tf.get_variable(
            'bias', 
            shape=[self.num_targets],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer
        )
        prediction = tf.matmul(features, w) + b
        
        loss = tf.reduce_mean(tf.square(targets-prediction))
        loss = tf.add(loss, tf.multiply(10.0, tf.global_norm([w])))
        
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.reduce_sum(tf.square(targets-tf.reshape(prediction, [-1])))
        
        return features, targets, train_op, eval_metric_ops
