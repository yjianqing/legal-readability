import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data import minibatches
from visualizer import print_example, create_viz_doc, visualize_example, write_viz_doc

class RNN(object):
    def __init__(self, hidden_size, keep_prob, rnn_cell_type, seed=None):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_type = rnn_cell_type
        self.seed = seed
        self.rnn_cell = self.rnn_cell_type(self.hidden_size)
        self.rnn_cell = tf.contrib.rnn.DropoutWrapper(self.rnn_cell, input_keep_prob=self.keep_prob, seed=self.seed)
    
    def build_graph(self, inputs, input_lengths):
        with tf.variable_scope("RNN"):
            output, state = tf.nn.dynamic_rnn(self.rnn_cell, inputs, input_lengths, dtype=tf.float32)
            #output = tf.nn.dropout(output, self.keep_prob, seed=self.seed)
            if self.rnn_cell_type is tf.nn.rnn_cell.LSTMCell:
                state = state[1] #just keep the h output
            return output, state


class BiRNN(object):
    def __init__(self, hidden_size, keep_prob, rnn_cell_type, seed=None):
        self.hidden_size = hidden_size / 2
        self.keep_prob = keep_prob
        self.rnn_cell_type = rnn_cell_type
        self.seed = seed
        self.rnn_cell_fw = self.rnn_cell_type(self.hidden_size)
        self.rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob, seed=self.seed)
        self.rnn_cell_bw = self.rnn_cell_type(self.hidden_size)
        self.rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob, seed=self.seed)
    
    def build_graph(self, inputs, input_lengths):
        with tf.variable_scope("RNN"):
            (fw_output, bw_output), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lengths, dtype=tf.float32)
            output = tf.concat([fw_output, bw_output], axis=2)
            #output = tf.nn.dropout(output, self.keep_prob, seed=self.seed)
            if self.rnn_cell_type is tf.nn.rnn_cell.LSTMCell:
                fw_state = fw_state[1] #just keep the h output
                bw_state = bw_state[1]
            state = tf.concat([fw_state, bw_state], axis=1)
            return output, state


class Classifier(object):
    def __init__(self, hidden_size, output_size, seed=None):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seed = seed
    
    def build_graph(self, inputs, reuse=False):
        with tf.variable_scope("Classifier", reuse=reuse):
            W = tf.get_variable("W", shape=(self.hidden_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            b = tf.get_variable("b", shape=(self.output_size), initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            output = tf.matmul(inputs, W) + b
            output = tf.nn.tanh(output)
            return output, W, b #un-normalized output


class Model(object):
    def __init__(self, word2id, embed_matrix, hidden_size, lr, dropout, rnn_cell_type, batch_size, epochs, seed=None):
        self.word2id = word2id
        self.embed_matrix = embed_matrix
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.rnn_cell_type = rnn_cell_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        with tf.variable_scope("Model"):
            self.add_placeholders()
            self.add_embedding(embed_matrix)
            self.build_graph()
            self.add_loss()
        self.add_train_op()
    
    def add_placeholders(self):
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="input_lengths")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="keep_prob")
    
    def add_embedding(self, embed_matrix):
        with tf.variable_scope("Embeddings"):
            tf_embed_matrix = tf.constant(embed_matrix, dtype=tf.float32, name="tf_embed_matrix")
            self.input_embeddings = tf.nn.embedding_lookup(tf_embed_matrix, self.input_ids)
    
    def build_graph(self):
        encoder = BiRNN(self.hidden_size, self.keep_prob, self.rnn_cell_type, seed=self.seed)
        self.hidden_states, final_state = encoder.build_graph(self.input_embeddings, self.input_lengths)
        classifier = Classifier(self.hidden_size, 2, seed=self.seed)
        self.result_logits, self.classifier_weights, self.classifier_bias = classifier.build_graph(final_state)
        seq_logits, _, _ = classifier.build_graph(tf.reshape(self.hidden_states, [-1, self.hidden_size]), reuse=True)
        batch_shape = tf.shape(self.hidden_states)
        self.seq_logits = tf.reshape(seq_logits, [batch_shape[0], batch_shape[1], -1])
    
    def add_loss(self):
        with tf.variable_scope("Loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.result_logits, labels=self.labels)
            self.loss = tf.reduce_mean(loss)
            #_, variance = tf.nn.moments(self.hidden_states, [1,2])
            #_, variance_h = tf.nn.moments(self.hidden_states, [2])
            #self.variance = 0.01 / tf.reduce_mean(variance) #+ tf.reduce_mean(variance_h))
            tf.summary.scalar('loss', self.loss)
    
    def preprocess_data(self, data):
        #convert string data into tokenized indexes
        max_tokens = 0
        result = []
        lengths = []
        for line in data:
            tokens = str(line).split()
            row = []
            if len(tokens) > max_tokens:
                max_tokens = len(tokens)
            for token in tokens:
                if token in self.word2id:
                    row.append(self.word2id[token])
                else:
                    row.append(self.word2id['$UNK'])
            result.append(row)
            lengths.append(len(tokens))
        #check batch max length and do padding
        for row in result:
            while len(row) < max_tokens:
                row.append(self.word2id['$PAD'])
        return np.array(result), np.array(lengths)
    
    def create_feed_dict(self, inputs_batch, input_lengths_batch, labels_batch=None, keep_prob=1.0):
        feed_dict = {}
        feed_dict[self.input_ids] = inputs_batch
        feed_dict[self.input_lengths] = input_lengths_batch
        if labels_batch is not None:
            feed_dict[self.labels] = labels_batch
        feed_dict[self.keep_prob] = keep_prob
        return feed_dict
    
    def add_train_op(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) #+ self.variance)
    
    def predict_batch(self, sess, inputs_batch):
        input_keys_batch, input_lengths_batch = self.preprocess_data(inputs_batch)
        feed_dict = self.create_feed_dict(input_keys_batch, input_lengths_batch)
        pred_scores, hidden_states, classifier_weights, classifier_bias, seq_scores = sess.run([self.result_logits, self.hidden_states, self.classifier_weights, self.classifier_bias, self.seq_logits], feed_dict=feed_dict)
        #Normalize sentence prediction scores
        pred_scores = tf.nn.softmax(pred_scores)
        #Modify classification weights into regression weights
        class_weights_0, class_weights_1 = tf.split(classifier_weights, num_or_size_splits=2, axis=1)
        classifier_weights = tf.squeeze(class_weights_1 - class_weights_0)
        classifier_bias = (classifier_bias[1] - classifier_bias[0]) / self.hidden_size #average out bias over hidden units
        #Weight hidden unit scores by their classification importance
        hidden_scores = hidden_states * classifier_weights + classifier_bias
        #Modify sequence classification probabilities into regression scores
        seq_scores = tf.nn.softmax(seq_scores)
        seq_scores_0, seq_scores_1 = tf.split(seq_scores, num_or_size_splits=2, axis=2)
        seq_scores = seq_scores_1 - seq_scores_0
        return pred_scores, hidden_scores, seq_scores
    
    def train_batch(self, sess, inputs_batch, labels_batch):
        input_keys_batch, input_lengths_batch = self.preprocess_data(inputs_batch)
        feed_dict = self.create_feed_dict(input_keys_batch, input_lengths_batch, labels_batch, 1.0 - self.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
    
    def predict(self, sess, eval_set, batch_random_seed=None, viz=None):
        def unpack(a, b):
            return a, b
        correct = 0
        total = 0
        total_batches = np.ceil(len(eval_set) / self.batch_size)
        viz_doc = create_viz_doc()
        with tqdm(unit="batches", total=total_batches) as pbar:
            for minibatch in minibatches(eval_set, self.batch_size, random_seed=batch_random_seed):
                inputs, labels = unpack(*minibatch)
                pred_scores, hidden_scores, seq_scores = self.predict_batch(sess, inputs)
                pred_scores = pred_scores.eval()
                hidden_scores = hidden_scores.eval()
                seq_scores = seq_scores.eval()
                predictions = np.argmax(pred_scores, axis=1)
                for n, pair in enumerate(zip(predictions, labels)):
                    if pair[0] == pair[1]:
                        correct += 1
                    elif viz == 'e':
                        #print_example(inputs[n], seq_scores[n].eval(), pred_scores[n], pair[1])
                        visualize_example(viz_doc, inputs[n], seq_scores[n], hidden_scores[n], pred_scores[n][1] - pred_scores[n][0], pair[1])
                    if viz == 's' and n % 1000 == 0:
                        visualize_example(viz_doc, inputs[n], seq_scores[n], hidden_scores[n], pred_scores[n][1] - pred_scores[n][0], pair[1])
                total += len(labels)
                accuracy = correct / total
                pbar.set_postfix(accuracy=accuracy)
                pbar.update()
        write_viz_doc(viz_doc)
        return accuracy
    
    def fit(self, sess, train_set, dev_set=None, batch_random_seed=None):
        for epoch in range(self.epochs):
            total_batches = np.ceil(len(train_set) / self.batch_size)
            with tqdm(unit="batches", total=total_batches) as pbar:
                for minibatch in minibatches(train_set, self.batch_size, random_seed=batch_random_seed):
                    loss = self.train_batch(sess, *minibatch)
                    pbar.set_postfix(loss=loss)
                    pbar.update()
            #dev_score = self.predict(sess, dev_set)
