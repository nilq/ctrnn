pub mod ctrnn;

#[cfg(test)]
mod tests {
    #[test]
    fn test_new_ctrnn() {
        use crate::ctrnn::CTRNN;
        use arrayfire::{Array, Dim4, col, row, print};

        let mut net = CTRNN::new(2, 0.1);

        for _ in 0..10 {
            net.euler_step(
                Array::new(&[10.0f32, 10.0f32], Dim4::new(&[2, 1, 1, 1]))
            )
        }

        print(net.output());
    }
}
