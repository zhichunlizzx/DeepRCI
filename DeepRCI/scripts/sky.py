from XMM import xmm1
import tensorflow as tf

model = xmm1()
model.build(input_shape=(None, 400, 400, 4))

model.load_weights('model.ckpt')


