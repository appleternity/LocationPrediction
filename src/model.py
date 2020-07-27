import tensorflow as tf
from tensorflow import layers
import numpy as np

class HAttention(object):
    def __init__(self, conf):
        self.conf = conf
        
        # output
        self.output = None
        self.predict = None
        self.acc = None
        self.result = None
        self.loss = None

    def build_model(self): 
        # input
        self.input_text = tf.placeholder(tf.int32, [None, self.conf.max_len], name="intput_text")
        self.input_char = tf.placeholder(tf.int32, [None, self.conf.max_char_len], name="intput_char")
        self.input_dropout_rate = tf.placeholder(tf.float32, name="input_dropout_rate")
        self.input_label = tf.placeholder(tf.int32, [None], name="input_label")
        self.input_country_label = tf.placeholder(tf.int32, [None], name="input_country_label")
        self.keep_rate = 1-self.input_dropout_rate

        # for analysis
        self.weight_list = []
        self.char_weights = {}

        # self-attention model
        text_rep = self.embedding(self.input_text, vocab_size=self.conf.vocab_size, dim=self.conf.emb_dim, zero_pad=True)

        # position encoding
        text_rep += self.embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(text_rep)[1]), 0), [tf.shape(text_rep)[0], 1]),
                                      vocab_size=self.conf.max_len, 
                                      dim=self.conf.hidden_dim,
                                      scope="enc_pe",
                                      zero_pad=False)

        for l in range(1, self.conf.layer_num+1):
            text_rep, weights = self.multihead_attention(text_rep, text_rep, dim=self.conf.hidden_dim, num_heads=self.conf.num_head, scope="m{}".format(l))
            text_rep = self.feedforward(text_rep, dims=[4*self.conf.hidden_dim, self.conf.hidden_dim], scope="f{}".format(l))
            self.weight_list.append(weights)

        # sentence-attention
        text_rep = tf.reduce_sum(text_rep, axis=1)
        print("text_rep shape = ", text_rep.get_shape())

        # char part
        char_rep = self.embedding(self.input_char, vocab_size=self.conf.char_size, dim=self.conf.char_dim, zero_pad=True, scope="char_emb")
        char_rep += self.embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(char_rep)[1]), 0), [tf.shape(char_rep)[0], 1]),
                                      vocab_size=self.conf.max_char_len, 
                                      dim=self.conf.char_hidden_dim,
                                      scope="enc_pe_char",
                                      zero_pad=False)
        
        # run cnn
        char_rep_list = self.cnn(char_rep, "char_cnn")

        # run self-attention
        all_tensors = [text_rep]
        for kernel_size, char_rep in char_rep_list:
            dim = self.conf.filter_list[kernel_size]

            for l in range(self.conf.char_layer_num, 1):
                char_rep, weight = self.multihead_attention(char_rep, char_rep, dim=dim, num_heads=self.conf.char_num_head, scope="char_k_{}_m{}".format(kernel_size, l))
                char_rep = self.feedforward(char_rep, dims=[4*dim, dim], scope="char_k{}_f{}".format(kernel_size, l))
                self.char_weights["{}_1".format(kernel_size)] = weight

            char_rep = tf.reduce_sum(char_rep, axis=1)
            all_tensors.append(char_rep)

        text_rep = tf.concat(all_tensors, axis=-1)
        
        print("text_rep+char_rep shape = ", text_rep.get_shape())
        
        # classifier
        if self.conf.reg:
            reg = tf.contrib.layers.l2_regularizer(scale=self.conf.reg_weight)
        else:
            reg = None

        # city
        output = text_rep
        output = tf.layers.dense(output, self.conf.output_size, kernel_regularizer=reg)
        
        # country
        country_output = text_rep
        country_output = tf.layers.dense(country_output, self.conf.country_output_size, kernel_regularizer=reg)
       
        # loss
        self.city_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.input_label)
        self.city_loss = tf.reduce_mean(self.city_loss)
        self.country_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=country_output, labels=self.input_country_label)
        self.country_loss = tf.reduce_mean(self.country_loss)
        self.loss = self.city_loss + self.country_loss

        if self.conf.reg:
            self.loss += tf.losses.get_regularization_loss()

        self.optim = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate).minimize(self.loss)

        # prediction
        self.output = tf.nn.softmax(output)
        self.predict = tf.cast(tf.argmax(self.output, 1), tf.int32)
        correct_pred = tf.equal(self.input_label, self.predict)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.result = correct_pred

        self.country_output = tf.nn.softmax(country_output)
        self.country_predict = tf.cast(tf.argmax(self.country_output, 1), tf.int32)
        
        # save
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    
    def cnn(self, vector, name):
        with tf.name_scope(name):
            if self.conf.reg:
                reg = tf.contrib.layers.l2_regularizer(scale=self.conf.reg_weight)
            else:
                reg = None

            outputs = {}
            for kernel_size, filter_num in self.conf.filter_list.items():
                conv = layers.conv1d(
                    vector, 
                    filter_num,
                    kernel_size,
                    activation=tf.nn.relu,
                    kernel_regularizer=reg,
                    name="cnn_k{}_f{}".format(kernel_size, filter_num)
                )

                res = tf.layers.max_pooling1d(
                    conv,
                    kernel_size,
                    kernel_size-1,
                    name="max_pooling_k{}_f{}".format(kernel_size, filter_num)
                )
                res = tf.nn.dropout(res, self.keep_rate)
                res = self.normalize(res)
                outputs[kernel_size] = res

        outputs = sorted(outputs.items(), key=lambda x: x[0])

        return outputs

    def positional_encoding(self, inputs, dim, zero_pad=False, scope="positional_encoding"):
        print(inputs.get_shape())
        #N, T = inputs.get_shape().as_list()
        #N = self.conf.batch_size
        T = inputs.get_shape()[1]
        with tf.variable_scope(scope):
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])
        
            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2.*i/dim) for i in range(dim)]
                for pos in range(T)], dtype=np.float32)
        
            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        
            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)
        
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, dim]),
                                          lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        
            return outputs

    def normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs

    def embedding(self, inputs, vocab_size, dim, scope="embedding", zero_pad=False):
        with tf.variable_scope(scope):
            if self.conf.reg:
                reg = tf.contrib.layers.l2_regularizer(scale=self.conf.reg_weight)
            else:
                reg = None
            lookup_table = tf.get_variable(
                'lookup_table',
                dtype=tf.float32,
                shape=[vocab_size, dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                #regularizer=reg
            )
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, dim]), lookup_table), 0)
            
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)
            
        return outputs

    def multihead_attention(self, queries, keys, dim=None, num_heads=8, scope="multihead_attention"):
        
        with tf.variable_scope(scope):
            # Set the fall back option for dim
            if dim is None:
                dim = queries.get_shape().as_list[-1]
            
            # Linear projections
            if self.conf.reg:
                reg = tf.contrib.layers.l2_regularizer(scale=self.conf.reg_weight)
            else:
                reg = None
            Q = tf.layers.dense(queries, dim, activation=tf.nn.relu, kernel_regularizer=reg) # (N, T_q, C)
            K = tf.layers.dense(keys, dim, activation=tf.nn.relu, kernel_regularizer=reg) # (N, T_k, C)
            V = tf.layers.dense(keys, dim, activation=tf.nn.relu, kernel_regularizer=reg) # (N, T_k, C)

            print(queries.get_shape())
            print(Q.get_shape())

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
    
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
             
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
              
            # Dropouts
            outputs = tf.nn.dropout(outputs, self.keep_rate)
            weights = outputs

            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
            weights = tf.concat(tf.split(weights, num_heads, axis=0), axis=2)

            # Residual connection
            outputs += queries
                  
            # Normalize
            outputs = self.normalize(outputs) # (N, T_q, C)
     
        return outputs, weights

    def feedforward(self, inputs, dims, scope):
        with tf.variable_scope(scope):
            if self.conf.reg:
                reg = tf.contrib.layers.l2_regularizer(scale=self.conf.reg_weight)
            else:
                reg = None

            outputs = tf.layers.dense(inputs, dims[0], activation=tf.nn.relu, kernel_regularizer=reg)
            outputs = tf.layers.dense(outputs, dims[1], kernel_regularizer=reg)
            outputs += inputs
            outputs = self.normalize(outputs)
            return outputs

            # Inner layer
            params = {"inputs": inputs, "filters": dims[0], "kernel_size": 1,
                    "activation": tf.nn.relu, "use_bias": True, "kernel_regularizer":reg}
            outputs = tf.layers.conv1d(**params)
            
            # Readout layer
            params = {"inputs": outputs, "filters": dims[1], "kernel_size": 1,
                    "activation": None, "use_bias": True, "kernel_regularizer":reg}
            outputs = tf.layers.conv1d(**params)
            
            # Residual connection
            outputs += inputs
            
            # Normalize
            outputs = self.normalize(outputs)
    
        return outputs

if __name__ == "__main__":
    pass
