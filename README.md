# XorCNTKcpp
XOR dataset on a simple MLP with CNTK2.7 in C++ and GPU with CUDA10.0

Assuming you have the CNTK2.7-GPU distribution in C:\cntk-gpu-2.7
Then you can build the project.

The Output :

```
XOR dataset CNTK!!! Device : GPU[0] GeForce 930MX

Start Training File ...
Minibatch Epoch:     0    loss = 0.704330    acc = 0.50
Minibatch Epoch:    50    loss = 0.673318    acc = 0.50
Minibatch Epoch:   100    loss = 0.604436    acc = 0.50
Minibatch Epoch:   150    loss = 0.491982    acc = 0.50
Minibatch Epoch:   200    loss = 0.321270    acc = 1.00
Minibatch Epoch:   250    loss = 0.134061    acc = 1.00
Minibatch Epoch:   300    loss = 0.066452    acc = 1.00
Minibatch Epoch:   350    loss = 0.040356    acc = 1.00
Minibatch Epoch:   400    loss = 0.027866    acc = 1.00
Minibatch Epoch:   450    loss = 0.020855    acc = 1.00
Minibatch Epoch:   500    loss = 0.016466    acc = 1.00
Minibatch Epoch:   550    loss = 0.013498    acc = 1.00
Minibatch Epoch:   600    loss = 0.011376    acc = 1.00
Minibatch Epoch:   650    loss = 0.009792    acc = 1.00
Minibatch Epoch:   700    loss = 0.008569    acc = 1.00
Minibatch Epoch:   750    loss = 0.007601    acc = 1.00
Minibatch Epoch:   800    loss = 0.006816    acc = 1.00
Minibatch Epoch:   850    loss = 0.006168    acc = 1.00
Minibatch Epoch:   900    loss = 0.005626    acc = 1.00
Minibatch Epoch:   950    loss = 0.005166    acc = 1.00
Minibatch Epoch:  1000    loss = 0.004771    acc = 1.00
End Training File ...

Prediction
[0 0] = 0 ~ 0.001827
[0 1] = 1 ~ 0.994444
[1 0] = 1 ~ 0.995401
[1 1] = 0 ~ 0.007024
```
