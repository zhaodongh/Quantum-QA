import theano
import theano.tensor as T 
from theano.tensor.nnet import conv
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample
import numpy as np

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie
    
    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie
    
    """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # self.y_score = T.nnet.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.y_score = self.p_y_given_x

        # parameters of the model
        self.params = [self.W, self.b]
        self.test = 0


    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

    .. math::
    
    \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
    \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
    \ell (\theta=\{W,b\}, \mathcal{D})
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    
    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # return -(T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + 10 * y * T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]))
        # print y.dtype
        

        self.test = T.log(self.p_y_given_x)

        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + 0.1 * ((w1 ** 2).sum() + \
        #     (w2 ** 2).sum() + (w2 ** 2).sum() + (w2 ** 2).sum())
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # return -T.max(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            # return T.arange(y.shape[0])
            # return self.p_y_given_x
            # return y
            # return T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
            # [T.arange(y.shape[0]), y]
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



class cnn_net(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize = (1, 1)):
        
        # assert image_shape[1] == filter_shape[1]  
        self.input = input  

        fan_in = np.prod(filter_shape[1:])#连乘 
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /  
                   np.prod(poolsize))  

        # print filter_shape[0]
        # print filter_shape[2:]
        # print poolsize
        # exit()

        W_bound = np.sqrt(6. / (fan_in + fan_out)) 
        # print W_bound
        # print fan_in
        # print fan_out
        # exit()
        self.W = theano.shared(  
            np.asarray(  
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),  
                dtype=theano.config.floatX  
            ),  
            borrow=True  
        )  

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)  
        self.b = theano.shared(value=b_values, borrow=True)  

        conv_out = conv.conv2d(  
            input=input,  
            filters=self.W,  
            filter_shape=filter_shape,  
            image_shape=image_shape, 
            # border_mode='full' 
        )  

    
        pooled_out = downsample.max_pool_2d(  
            input=conv_out,  
            ds=poolsize,  
            ignore_border=True  
        )  
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))  
        # self.output = ReLU(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))  
        # self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')  

        self.params = [self.W, self.b]  

class Density(object):
    def __init__(self, rng, input, in_len, max_len):
       
        smooth_term = theano.shared(value= np.asarray(0.0001, dtype=theano.config.floatX))
        # smooth_term = theano.shared(0.0001)
        self.W = theano.shared(
                value= np.ones(max_len, dtype=theano.config.floatX),
                name='W')
        W_effection = self.W
        self.test1 = in_len.repeat(input.shape[1])
        self.test2 = W_effection.repeat(input.shape[0])\
            .reshape((input.shape[1], input.shape[0])).T.flatten()

        projectors, _ = theano.scan(fn = lambda projector, coef, somooth: coef * \
            T.outer(projector, projector) / (T.dot(projector, projector) + somooth) , \
            outputs_info = None, sequences = [input.reshape((input.shape[0] * \
                input.shape[1], input.shape[2])), W_effection.repeat(input.shape[0])\
            .reshape((input.shape[1], input.shape[0])).T.flatten()], \
            non_sequences = [smooth_term])
        '''
sequences = [
input.reshape((input.shape[0] * input.shape[1], input.shape[2])), 
W_effection.repeat(input.shape[0]).reshape((input.shape[1], input.shape[0])).T.flatten()]
        '''
        self.test = projectors
        self.output = T.tanh(T.sum(projectors.reshape((input.shape[0], input.shape[1], \
            input.shape[2], input.shape[2])), axis = 1))

        self.params = [self.W]
        # self.params = []

class Density_Dot(object):
    def __init__(self, rng, input1, input2):
        # assert input1.shape[0] == input2.shape[0]

        # W_bound = np.sqrt(6. / (fan_in + fan_out)) 

        densities, _ = theano.scan(fn = lambda density1, density2 : \
            T.dot(density1, density2), outputs_info = None, \
            sequences = [input1, input2])

        # densities = T.mul(input1, input2)


        self.output = densities
def share_x(data_xs):
    res = []
    for data_x in data_xs:
        res.append(theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow = True))
    return res

def share_y(data_x):
    return T.cast(theano.shared(np.asarray(data_x), borrow = True), 'int32')
        








