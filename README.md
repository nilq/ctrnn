# 🐘 CTRNN

A super tiny continuous-time recurrent neural network using ArrayFire.

## Usage

The idea would be to use this with a separate genetic algorithm setup.

```rs
use ctrnn::CTRNN;

let net = CTRNN::new(2, 0.1);
net.euler_step();
```
