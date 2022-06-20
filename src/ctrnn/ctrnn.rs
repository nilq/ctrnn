use super::util::inverse_sigmoid;
use arrayfire::{ Array, constant, Dim4, randu, matmul, MatProp, sigmoid };


pub struct CTRNN {
    size: u64,
    step_size: f32,

    taus: Array<f32>,
    bias: Array<f32>,
    gains: Array<f32>,
    weights: Array<f32>,
    states: Array<f32>,
    output: Array<f32>
}


impl CTRNN {
    pub fn new(size: u64, step_size: f32) -> Self {
        let array_dims = Dim4::new(&[size, 1, 1, 1]);

        CTRNN {
            size,
            step_size,

            taus: constant(1.0f32, array_dims),
            bias: constant(1.0f32, array_dims),
            gains: constant(1.0f32, array_dims),
            weights: randu::<f32>(Dim4::new(&[size, size, 1, 1])),
            states: randu::<f32>(array_dims),
            output: constant(0.0f32, array_dims),
        }
    }

    pub fn euler_step(&mut self, inputs: Array<f32>) {
        let total = inputs + matmul(&self.weights, &self.output, MatProp::NONE, MatProp::NONE);
 
        self.set_states(&self.states + self.step_size * (1.0f32 / &self.taus) * (total - &self.states));        
        self.set_output(sigmoid(&(&self.gains * &(&self.states + &self.bias))))
    }


    pub fn output(&self) -> &Array<f32> {
        &self.output
    }


    fn set_output(&mut self, output: Array<f32>) {
        self.output = output;
        self.states = inverse_sigmoid(&self.output) / &self.gains - &self.bias
    }

    fn set_states(&mut self, states: Array<f32>) {
        self.states = states;
        self.output = sigmoid(&self.states)
    }
}