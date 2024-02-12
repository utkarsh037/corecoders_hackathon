from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Add these imports for Intel oneAPI libraries
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["OMP_NUM_THREADS"] = "1"

# Specify the path to Intel TensorFlow
import intel-tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
from tensorflow.python.compiler.tensorrt import trt_convert as trt

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('homepage.html')

# ... (other routes remain the same)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    data1 = int(float(request.form['a']))
    data2 = int(float(request.form['b']))
    data3 = int(float(request.form['c']))
    print(data1, data2, data3)
    arr = np.array([[data1, data2, data3]])

    # Use Intel TensorFlow for prediction
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        output = sess.run(model, feed_dict={input_tensor: arr})

    def to_str(var):
        return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

    if output < 4:
        return render_template('prediction.html', p=to_str(output), q=' No ')
    elif 4 <= output < 6:
        return render_template('prediction.html', p=to_str(output), q=' Low ')
    elif 6 <= output < 8:
        return render_template('prediction.html', p=to_str(output), q=' Moderate ')
    elif 8 <= output < 9:
        return render_template('prediction.html', p=to_str(output), q=' High ')
    elif output >= 9:
        return render_template('prediction.html', p=to_str(output), q=' Very High ')
    else:
        return render_template('prediction.html', p=' N.A.', q=' Undefined ')

if __name__ == "__main__":
    app.run(debug=True)
