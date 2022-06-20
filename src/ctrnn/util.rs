use arrayfire::{ Array, log };

pub fn inverse_sigmoid(v: &Array<f32>) -> Array<f32> {
    log(&(v / &(1.0f32 / v)))
}