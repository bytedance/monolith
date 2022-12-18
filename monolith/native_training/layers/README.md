The layers in Monolith are a super set of tensorflow keras layers. 
Monolith adds/enhances the following layers:
- Dense
- MLP
- AddBias
- LayerNorm/GradNorm
- GroupInt/AllInt/CDot/CAN/DCN/CIN/AutoInt/SeNet/iRazor/DIN/DIEN/DMR_U2I
- LogitCorrection
- SumPooling/AvgPooling/MaxPooling

Monolith layers are compatible with keras layers, that means you can mix usage of keras layers and monolith layers. 
Here is an example of creating monolith layer:
```python
import tensorflow as tf
from monolith.native_training.layers import Dense

# the first method to new a monolith layer, which is the same as keras
dense = Dense(units=100, activation=tf.keras.activations.relu)

# the second method to new a monlith layer
dense_p = Dense.params()
dense_p.units=100
dense_p.activation=tf.keras.activations.relu
dense2 = dense_p.instantiate()

model = tf.keras.Sequential([
  dense,    # create from constructor
  dense2,   # create from new_instance
  tf.keras.layers.Dense(units=100, activation=tf.keras.activations.relu)  # mix use
])
```
As show above, there is two methods to create a layer, one is using constructor, 
the other employ `new_instance` method of `InstantiableParams`. 

In most case, just replace:
```python
from tensorflow.keras import layers
```
with 
```python
from monolith.native_training import layers
```

we prefer monolith layers. 
