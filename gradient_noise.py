
"""
This performs the incorporation of noise to the gradients of the network parameters 
based on the implementation of the following paper:
 https://arxiv.org/abs/1511.06807
 and https://github.com/cpury/keras_gradient_noise/blob/master/keras_gradient_noise/gradient_noise.py
 while modification of the std of the noise to be proportinal to the learning rate
"""
import inspect
import importlib
import tensorflow as tf 

##  to adjust the variance as the learning rate 
def add_gradient_noise(BaseOptimizer, keras=None):
    """
    This class returns a modified keras optimizer that
    supports adding gradient noise to the gradients as described in this paper:
    https://arxiv.org/abs/1511.06807
    alpha is a pre-defiend parameter, which is selected by the user. The std of the noise is lr/alpha
    """
    if keras is None:
        # Import it automatically. Try to guess from the optimizer's module
        if hasattr(BaseOptimizer, '__module__') and BaseOptimizer.__module__.startswith('keras'):
            keras = importlib.import_module('keras')
        else:
            keras = importlib.import_module('tensorflow.keras')

    K = keras.backend

    if not (
        inspect.isclass(BaseOptimizer) and
        issubclass(BaseOptimizer, keras.optimizers.Optimizer)
    ):
        raise ValueError(
            'add_gradient_noise() expects a valid Keras optimizer'
        )

    def _get_shape(x):
        if hasattr(x, 'dense_shape'):
            return x.dense_shape

        return K.shape(x)

    class NoisyOptimizer(BaseOptimizer):
        def __init__(self, alpha, **kwargs):
            super(NoisyOptimizer, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.alpha = K.variable(alpha, name='alpha')
               
    
        def get_gradients(self, loss, params):
            grads = super(NoisyOptimizer, self).get_gradients(loss, params)
            LR = super(NoisyOptimizer, self).lr
            #t = self.iterations

          
            #if tf.math.less(t, 400) is not None:
                ###### 1
            variance = K.cast(LR, K.dtype(grads[0]))
            variance = variance/self.alpha # 
            ######### 2
            #t = K.cast(self.iterations, K.dtype(grads[0]))
            #variance = variance / ((1 + t) ** 0.55)
            #else :
             #   variance = K.cast(0.001, K.dtype(grads[0]))
                

            grads = [
                    grad + K.random_normal(
                    _get_shape(grad),
                    mean=0.0,
                    stddev=variance,
                    dtype=K.dtype(grads[0])
                )
                for grad in grads
            ]

            return grads

        def get_config(self):
            config = {'alpha': float(K.get_value(self.alpha))}
            base_config = super(NoisyOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    NoisyOptimizer.__name__ = 'Noisy{}'.format(BaseOptimizer.__name__)

    return NoisyOptimizer
