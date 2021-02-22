from keras import backend as K
# Keras는 텐서 곱셈, 합성 곱 등의 저수준의 연산을 제공하지 않고 대신 Keras의 "백엔드 엔진"역할을하는 특수하고 잘 최적화 된 텐서 라이브러리를 사용
from keras.layers import Conv2D

# https://github.com/atriumlts/subpixel/blob/master/keras_subpixel.py

"""
	Subpixel Layer as a child class of Conv2D. This layer accepts all normal
	arguments, with the exception of dilation_rate(). The argument r indicates
	the upsampling factor, which is applied to the normal output of Conv2D.
	The output of this layer will have the same number of channels as the
	indicated filter field, and thus works for grayscale, color, or as a a
	hidden layer.
	Arguments:
		*see Keras Docs for Conv2D args, noting that dilation_rate() is removed*
		r: upscaling factor, which is applied to the output of normal Conv2D
	A test is included, which performs super-resolution on the Cifar10 dataset.
	Since these images are small, only a scale factor of 2 is used. Test images
	are saved in the directory 'test_output/'. This test runs for 5 epochs,
	which can be altered in line 132. You can run this test by using the
	following commands:
	mkdir test_output
	python keras_subpixel.py
"""

# 클래스 Subpixel(Conv2D)로 선언
class Subpixel(Conv2D):                                                                             
    # 클래스 기본 초기화
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    # r^2 channels로 만들어진 feature map들을 순서대로 배치하는 부분
    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)  permute_dimensions : 텐서의 축을 치환합니다. 
        #Keras backend does not support tf.split, so in future versions this could be nicer
        #케라스 백엔드는 tf.split를 지원하지 않습니다, 그래서 향후 버전에서는 더 좋아질 수 있습니다.
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config['filters'] = int(config['filters'] / (self.r*self.r))
        config['r'] = self.r
        return config