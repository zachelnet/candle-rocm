//! Smoke tests for the ROCm backend.
//! Run: ROCM_PATH=/usr HIP_PATH=/usr cargo test --features rocm --test rocm_smoke

#[cfg(feature = "rocm")]
mod rocm {
    use candle_core::{DType, Device, Result, Tensor};

    fn dev() -> Result<Device> { Device::new_rocm(0) }
    fn tensor(shape: &[usize], dev: &Device) -> Result<Tensor> { Tensor::ones(shape, DType::F32, dev) }

    #[test] fn affine() -> Result<()> {
        let d = dev()?; let x = tensor(&[2,3,4,4], &d)?;
        let cpu = Tensor::ones(&[2,3,4,4], DType::F32, &Device::Cpu)?.affine(2.0,1.0)?;
        let y: Vec<f32> = x.affine(2.0,1.0)?.flatten_all()?.to_vec1()?;
        let e: Vec<f32> = cpu.flatten_all()?.to_vec1()?;
        assert_eq!(y, e); Ok(())
    }
    #[test] fn add() -> Result<()> {
        let d = dev()?;
        assert_eq!((tensor(&[2,3], &d)? + tensor(&[2,3], &d)?)?.to_vec2::<f32>()?, vec![vec![2.0f32;3];2]); Ok(())
    }
    #[test] fn mul() -> Result<()> {
        let d = dev()?;
        assert_eq!((tensor(&[2,3], &d)? * tensor(&[2,3], &d)?)?.to_vec2::<f32>()?, vec![vec![1.0f32;3];2]); Ok(())
    }
    #[test] fn relu() -> Result<()> {
        assert_eq!(Tensor::new(&[-1.0f32,0.0,1.0], &dev()?)?.relu()?.to_vec1::<f32>()?, vec![0.0,0.0,1.0]); Ok(())
    }
    #[test] fn neg() -> Result<()> {
        assert_eq!(Tensor::new(&[1.0f32,-2.0], &dev()?)?.neg()?.to_vec1::<f32>()?, vec![-1.0,2.0]); Ok(())
    }
    #[test] fn sum_all() -> Result<()> {
        assert_eq!(tensor(&[2,3], &dev()?)?.sum_all()?.to_vec0::<f32>()?, 6.0); Ok(())
    }
    #[test] fn max_reduce() -> Result<()> {
        assert_eq!(Tensor::new(&[1.0f32,3.0,2.0], &dev()?)?.max_all()?.to_vec0::<f32>()?, 3.0); Ok(())
    }
    #[test] fn max_pool2d() -> Result<()> {
        let d = dev()?;
        let x = Tensor::arange(1f32, 17f32, &d)?.reshape((1,1,4,4))?;
        let y = x.max_pool2d_with_stride((2,2),(2,2))?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(y, vec![6.0,8.0,14.0,16.0]); Ok(())
    }
    #[test] fn upsample_nearest2d() -> Result<()> {
        let x = Tensor::new(&[1.0f32,2.0,3.0,4.0], &dev()?)?.reshape((1,1,2,2))?;
        let up = x.upsample_nearest2d(4,4)?; let dims = up.dims();
        assert_eq!(dims[2],4); assert_eq!(dims[3],4); Ok(())
    }
    #[test] fn cast_u8_to_f32() -> Result<()> {
        assert_eq!(Tensor::new(&[1u8,2,3], &dev()?)?.to_dtype(DType::F32)?.to_vec1::<f32>()?, vec![1.0,2.0,3.0]); Ok(())
    }
    #[test] fn conv2d() -> Result<()> {
        let d = dev()?;
        let x = Tensor::ones((1,1,4,4), DType::F32, &d)?;
        let w = Tensor::ones((1,1,3,3), DType::F32, &d)?;
        // (4-3)/1+1 = 2
        assert_eq!(x.conv2d(&w,0,1,1,1)?.dims()[2], 2); Ok(())
    }
    #[test] fn index_select() -> Result<()> {
        let d = dev()?;
        let x = Tensor::new(&[1.0f32,2.0,3.0,4.0], &d)?;
        assert_eq!(x.index_select(&Tensor::new(&[0u32,2], &d)?, 0)?.to_vec1::<f32>()?, vec![1.0,3.0]); Ok(())
    }
    #[test] fn elu() -> Result<()> {
        let d = dev()?;
        let x = Tensor::new(&[-1.0f32,0.0,1.0], &d)?;
        let (a,b) = (x.elu(1.0)?.to_vec1::<f32>()?, x.to_device(&Device::Cpu)?.elu(1.0)?.to_vec1::<f32>()?);
        for (r, e) in a.iter().zip(b.iter()) { assert!((r-e).abs() < 0.001); } Ok(())
    }
    #[test] fn bf16_ops() -> Result<()> {
        let d = dev()?;
        let x = Tensor::ones((2,3,4,4), DType::BF16, &d)?;
        let s = x.affine(2.0,1.0)?.add(&x)?.relu()?.sum_all()?;
        assert!((s.to_dtype(DType::F32)?.to_vec0::<f32>()? - 384.0).abs() < 1.0); Ok(())
    }
    #[test] fn bf16_max_pool() -> Result<()> {
        let d = dev()?;
        let x = Tensor::arange(0f32,16f32,&d)?.reshape((1,1,4,4))?.to_dtype(DType::BF16)?;
        let y = x.max_pool2d_with_stride((2,2),(2,2))?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        assert_eq!(y, vec![5.0,7.0,13.0,15.0]); Ok(())
    }
}
