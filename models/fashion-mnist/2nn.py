import tensorflow as tf

from model import Model


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        w1 = tf.Variable(
            tf.random_normal([IMAGE_SIZE * IMAGE_SIZE, 200]))
        b1 = tf.Variable(tf.random_normal([200]))
        w2 = tf.Variable(
            tf.random_normal([200, 200]))
        b2 = tf.Variable(tf.random_normal([200]))
        w3 = tf.Variable(
            tf.random_normal([200, self.num_classes]))
        b3 = tf.Variable(tf.random_normal([self.num_classes]))

        hidden1 = tf.nn.relu(tf.add(tf.matmul(features, w1), b1))
        hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, w2), b2))
        logits = tf.add(tf.matmul(hidden2, w3), b3)

        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, eval_metric_ops
