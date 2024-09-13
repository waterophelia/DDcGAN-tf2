import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

def deconv_ir(input_tensor, strides, scope_name):
    weight_vars = []
    scope = ['deconv1']
    with tf.name_scope('Generator'):
        with tf.name_scope(scope_name):
            weight_vars.append(_create_variables(1, 1, 3, scope=scope[0]))
    
    deconv_num = len(weight_vars)
    out = input_tensor
    for i in range(deconv_num):
        input_shape = tf.shape(out)
        kernel = weight_vars[i]

        # Calculate dynamic output shape correctly
        output_shape = [input_shape[0], input_shape[1] * strides[1], input_shape[2] * strides[2], input_shape[3]]

        out = tf.nn.conv2d_transpose(
            out,
            filters=kernel,
            output_shape=output_shape,
            strides=strides,
            padding='SAME'
        )
    return out

def deconv_vis(input_tensor, strides, scope_name):
    weight_vars = []
    scope = ['deconv1']
    with tf.name_scope('Generator'):
        with tf.name_scope(scope_name):
            weight_vars.append(_create_variables(1, 1, 3, scope=scope[0]))
    
    deconv_num = len(weight_vars)
    out = input_tensor
    for i in range(deconv_num):
        input_shape = tf.shape(out)
        kernel = weight_vars[i]

        # Calculate dynamic output shape correctly
        output_shape = [input_shape[0], input_shape[1] * strides[1], input_shape[2] * strides[2], input_shape[3]]

        out = tf.nn.conv2d_transpose(
            out,
            filters=kernel,
            output_shape=output_shape,
            strides=strides,
            padding='SAME'
        )
    return out

def _create_variables(input_filters, output_filters, kernel_size, scope):
    shape = [kernel_size, kernel_size, output_filters, input_filters]
    with tf.name_scope(scope):
        kernel = tf.Variable(tf.random.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
    return kernel

if __name__ == "__main__":
    # Create a dummy input tensor with a batch size of 1, height 8, width 8, and 1 channel
    dummy_input = tf.random.normal([1, 8, 8, 1])
    # Define strides
    strides = [1, 2, 2, 1]
    # Test the deconv_ir function
    output_ir = deconv_ir(dummy_input, strides, scope_name='deconv_ir_test')
    print("Output shape from deconv_ir:", output_ir.shape)
    # Test the deconv_vis function
    output_vis = deconv_vis(dummy_input, strides, scope_name='deconv_vis_test')
    print("Output shape from deconv_vis:", output_vis.shape)
