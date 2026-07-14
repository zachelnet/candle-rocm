// src/rocblas/level1.rs

use crate::hip::DeviceMemory;
use crate::*;
use crate::rocblas::bindings::_rocblas_handle;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;

//==============================================================================
// SCAL functions
//==============================================================================

/// Scale a vector by a scalar
///
/// x := alpha * x
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vector x
/// * `alpha` - Scalar
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
pub fn scal<T>(handle: &Handle, n: i32, alpha: &T, x: &DeviceMemory<T>, incx: i32) -> Result<()>
where
    T: ScalType,
{
    unsafe { T::rocblas_scal(handle, n, alpha, x.as_ptr().cast(), incx) }
}

/// Scale vectors in a batch by a scalar
///
/// x_i := alpha * x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i
/// * `alpha` - Scalar
/// * `x` - Device array of device pointers to each vector x_i
/// * `incx` - Stride between consecutive elements of each x_i
/// * `batch_count` - Number of instances in the batch
pub fn scal_batched<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *const *mut T,
    incx: i32,
    batch_count: i32,
) -> Result<()>
where
    T: ScalBatchedType,
{
    unsafe { T::rocblas_scal_batched(handle, n, alpha, x, incx, batch_count) }
}

/// Scale vectors in a strided batch by a scalar
///
/// x_i := alpha * x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i
/// * `alpha` - Scalar
/// * `x` - Device pointer to first vector x_1
/// * `incx` - Stride between consecutive elements of each x_i
/// * `stride_x` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `batch_count` - Number of instances in the batch
pub unsafe fn scal_strided_batched<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *mut T,
    incx: i32,
    stride_x: i64,
    batch_count: i32,
) -> Result<()>
where
    T: ScalStridedBatchedType,
{
    unsafe {
        T::rocblas_scal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
    }
}

//==============================================================================
// COPY functions
//==============================================================================

/// Copy a vector
///
/// y := x
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
pub unsafe fn copy<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: CopyType,
{
    unsafe { T::rocblas_copy(handle, n, x, incx, y, incy) }
}

/// Copy vectors in a batch
///
/// y_i := x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i and y_i
/// * `x` - Device array of device pointers to each vector x_i
/// * `incx` - Stride between consecutive elements of each x_i
/// * `y` - Device array of device pointers to each vector y_i
/// * `incy` - Stride between consecutive elements of each y_i
/// * `batch_count` - Number of instances in the batch
pub unsafe fn copy_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: CopyBatchedType,
{
    unsafe {
        T::rocblas_copy_batched(handle, n, x, incx, y, incy, batch_count)
    }
}

/// Copy vectors in a strided batch
///
/// y_i := x_i, for i = 1,...,batch_count
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in each vector x_i and y_i
/// * `x` - Device pointer to first vector x_1
/// * `incx` - Stride between consecutive elements of each x_i
/// * `stridex` - Stride from start of one vector (x_i) to the next (x_i+1)
/// * `y` - Device pointer to first vector y_1
/// * `incy` - Stride between consecutive elements of each y_i
/// * `stridey` - Stride from start of one vector (y_i) to the next (y_i+1)
/// * `batch_count` - Number of instances in the batch
pub unsafe fn copy_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    y: *mut T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
) -> Result<()>
where
    T: CopyStridedBatchedType,
{
    unsafe {
        T::rocblas_copy_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count)
    }
}

//==============================================================================
// DOT functions
//==============================================================================

/// Compute the dot product of two vectors
///
/// result := x * y
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
/// * `result` - Pointer to the result
pub unsafe fn dot<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotType,
{
    unsafe { T::rocblas_dot(handle, n, x, incx, y, incy, result) }
}

/// Compute the dot product of two complex vectors
///
/// result := x * y (non-conjugated dot product)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
/// * `result` - Pointer to the result
pub unsafe fn dotu<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotuType,
{
    unsafe { T::rocblas_dotu(handle, n, x, incx, y, incy, result) }
}

/// Compute the conjugated dot product of two complex vectors
///
/// result := conjugate(x) * y
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in vectors x and y
/// * `x` - Device pointer to vector x
/// * `incx` - Stride between consecutive elements of x
/// * `y` - Device pointer to vector y
/// * `incy` - Stride between consecutive elements of y
/// * `result` - Pointer to the result
pub unsafe fn dotc<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    y: *const T,
    incy: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotcType,
{
    unsafe { T::rocblas_dotc(handle, n, x, incx, y, incy, result) }
}

//==============================================================================
// Type traits for implementation
//==============================================================================

impl_rocblas_traits!(
    ScalType,
    ScalTypeFn,
    {
        f32 => ffi::rocblas_sscal,
        f64 => ffi::rocblas_dscal,
        ffi::rocblas_float_complex => ffi::rocblas_cscal,
        ffi::rocblas_double_complex => ffi::rocblas_zscal,
    },
    rocblas_scal,
    (handle: &Handle, n: i32, alpha: &Self, x: *mut Self, incx: i32),
    (*mut _rocblas_handle, i32, *const T, *mut T, i32),
    (handle.as_raw(), n, alpha, x, incx)
);

impl_rocblas_traits!(
    ScalBatchedType,
    ScalBatchedTypeFn,
    {
        f32 => ffi::rocblas_sscal_batched,
        f64 => ffi::rocblas_dscal_batched,
        ffi::rocblas_float_complex => ffi::rocblas_cscal_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zscal_batched,
    },
    rocblas_scal_batched,
    (handle: &Handle, n: i32, alpha: &Self, x: *const *mut Self, incx: i32, batch_count: i32),
    (*mut _rocblas_handle, i32, *const T, *const *mut T, i32, i32),
    (handle.as_raw(), n, alpha, x, incx, batch_count)
);

impl_rocblas_traits!(
    ScalStridedBatchedType,
    ScalStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_sscal_strided_batched,
        f64 => ffi::rocblas_dscal_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_cscal_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zscal_strided_batched,
    },
    rocblas_scal_strided_batched,
    (handle: &Handle, n: i32, alpha: &Self, x: *mut Self, incx: i32, stride_x: i64, batch_count: i32),
    (*mut _rocblas_handle, i32, *const T, *mut T, i32, i64, i32),
    (handle.as_raw(), n, alpha, x, incx, stride_x, batch_count)
);

impl_rocblas_traits!(
    CopyType,
    CopyTypeFn,
    {
        f32 => ffi::rocblas_scopy,
        f64 => ffi::rocblas_dcopy,
        ffi::rocblas_float_complex => ffi::rocblas_ccopy,
        ffi::rocblas_double_complex => ffi::rocblas_zcopy,
    },
    rocblas_copy,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, y: *mut Self, incy: i32),
    (*mut _rocblas_handle, i32, *const T, i32, *mut T, i32),
    (handle.as_raw(), n, x, incx, y, incy)
);

impl_rocblas_traits!(
    CopyBatchedType,
    CopyBatchedTypeFn,
    {
        f32 => ffi::rocblas_scopy_batched,
        f64 => ffi::rocblas_dcopy_batched,
        ffi::rocblas_float_complex => ffi::rocblas_ccopy_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zcopy_batched,
    },
    rocblas_copy_batched,
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, y: *const *mut Self, incy: i32, batch_count: i32,),
    (*mut _rocblas_handle, i32, *const *const T, i32, *const *mut T, i32, i32),
    (handle.as_raw(), n, x, incx, y, incy, batch_count)
);

impl_rocblas_traits!(
    CopyStridedBatchedType,
    CopyStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_scopy_strided_batched,
        f64 => ffi::rocblas_dcopy_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_ccopy_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zcopy_strided_batched,
    },
    rocblas_copy_strided_batched,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, y: *mut Self, incy: i32, stridey: i64, batch_count: i32),
    (*mut _rocblas_handle, i32, *const T, i32, i64, *mut T, i32, i64, i32),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count)
);

impl_rocblas_traits!(
    DotType,
    DotTypeFn,
    {
        f32 => ffi::rocblas_sdot,
        f64 => ffi::rocblas_ddot,
        ffi::rocblas_half => ffi::rocblas_hdot,
        ffi::rocblas_bfloat16 => ffi::rocblas_bfdot,
    },
    rocblas_dot,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, y: *const Self, incy: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const T, i32, *const T, i32, *mut T),
    (handle.as_raw(), n, x, incx, y, incy, result)
);

impl_rocblas_traits!(
    DotuType,
    DotuTypeFn,
    {
        ffi::rocblas_float_complex => ffi::rocblas_cdotu,
        ffi::rocblas_double_complex => ffi::rocblas_zdotu,
    },
    rocblas_dotu,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, y: *const Self, incy: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const T, i32, *const T, i32, *mut T),
    (handle.as_raw(), n, x, incx, y, incy, result)
);

impl_rocblas_traits!(
    DotcType,
    DotcTypeFn,
    {
        ffi::rocblas_float_complex => ffi::rocblas_cdotc,
        ffi::rocblas_double_complex => ffi::rocblas_zdotc,
    },
    rocblas_dotc,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, y: *const Self, incy: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const T, i32, *const T, i32, *mut T),
    (handle.as_raw(), n, x, incx, y, incy, result)
);

//==============================================================================
// AXPY functions
//==============================================================================

/// Compute y := alpha * x + y
pub unsafe fn axpy<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    y: *mut T,
    incy: i32,
) -> Result<()>
where
    T: AxpyType,
{
    unsafe { T::rocblas_axpy(handle, n, alpha, x, incx, y, incy) }
}

/// Compute y_i := alpha * x_i + y_i, for i = 1,...,batch_count
pub unsafe fn axpy_batched<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *const *const T,
    incx: i32,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: AxpyBatchedType,
{
    unsafe { T::rocblas_axpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) }
}

/// Compute y_i := alpha * x_i + y_i, for i = 1,...,batch_count (strided)
pub unsafe fn axpy_strided_batched<T>(
    handle: &Handle,
    n: i32,
    alpha: &T,
    x: *const T,
    incx: i32,
    stridex: i64,
    y: *mut T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
) -> Result<()>
where
    T: AxpyStridedBatchedType,
{
    unsafe {
        T::rocblas_axpy_strided_batched(
            handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count,
        )
    }
}

//==============================================================================
// NRM2 functions
//==============================================================================

/// Compute the Euclidean norm of a vector: result := || x ||
pub unsafe fn nrm2<T, R>(handle: &Handle, n: i32, x: *const T, incx: i32, result: *mut R) -> Result<()>
where
    T: Nrm2Type<Result = R>,
{
    unsafe { T::rocblas_nrm2(handle, n, x, incx, result) }
}

/// Compute the Euclidean norm of each vector in a batch
pub unsafe fn nrm2_batched<T, R>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    batch_count: i32,
    result: *mut R,
) -> Result<()>
where
    T: Nrm2BatchedType<Result = R>,
{
    unsafe { T::rocblas_nrm2_batched(handle, n, x, incx, batch_count, result) }
}

/// Compute the Euclidean norm of each vector in a strided batch
pub unsafe fn nrm2_strided_batched<T, R>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    batch_count: i32,
    result: *mut R,
) -> Result<()>
where
    T: Nrm2StridedBatchedType<Result = R>,
{
    unsafe { T::rocblas_nrm2_strided_batched(handle, n, x, incx, stridex, batch_count, result) }
}

//==============================================================================
// ASUM functions
//==============================================================================

/// Compute the sum of the absolute values of a vector: result := sum(|x_i|)
pub unsafe fn asum<T, R>(handle: &Handle, n: i32, x: *const T, incx: i32, result: *mut R) -> Result<()>
where
    T: AsumType<Result = R>,
{
    unsafe { T::rocblas_asum(handle, n, x, incx, result) }
}

/// Compute the sum of the absolute values of each vector in a batch
pub unsafe fn asum_batched<T, R>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    batch_count: i32,
    result: *mut R,
) -> Result<()>
where
    T: AsumBatchedType<Result = R>,
{
    unsafe { T::rocblas_asum_batched(handle, n, x, incx, batch_count, result) }
}

/// Compute the sum of the absolute values of each vector in a strided batch
pub unsafe fn asum_strided_batched<T, R>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    batch_count: i32,
    result: *mut R,
) -> Result<()>
where
    T: AsumStridedBatchedType<Result = R>,
{
    unsafe { T::rocblas_asum_strided_batched(handle, n, x, incx, stridex, batch_count, result) }
}

//==============================================================================
// AMAX / AMIN functions
//==============================================================================

/// Find the index of the element with the maximum absolute value
pub unsafe fn amax<T>(handle: &Handle, n: i32, x: *const T, incx: i32, result: *mut i32) -> Result<()>
where
    T: AmaxType,
{
    unsafe { T::rocblas_amax(handle, n, x, incx, result) }
}

/// Find the index of the element with the maximum absolute value, for each vector in a batch
pub unsafe fn amax_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    batch_count: i32,
    result: *mut i32,
) -> Result<()>
where
    T: AmaxBatchedType,
{
    unsafe { T::rocblas_amax_batched(handle, n, x, incx, batch_count, result) }
}

/// Find the index of the element with the maximum absolute value, for each vector in a strided batch
pub unsafe fn amax_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    batch_count: i32,
    result: *mut i32,
) -> Result<()>
where
    T: AmaxStridedBatchedType,
{
    unsafe { T::rocblas_amax_strided_batched(handle, n, x, incx, stridex, batch_count, result) }
}

/// Find the index of the element with the minimum absolute value
pub unsafe fn amin<T>(handle: &Handle, n: i32, x: *const T, incx: i32, result: *mut i32) -> Result<()>
where
    T: AminType,
{
    unsafe { T::rocblas_amin(handle, n, x, incx, result) }
}

/// Find the index of the element with the minimum absolute value, for each vector in a batch
pub unsafe fn amin_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    batch_count: i32,
    result: *mut i32,
) -> Result<()>
where
    T: AminBatchedType,
{
    unsafe { T::rocblas_amin_batched(handle, n, x, incx, batch_count, result) }
}

/// Find the index of the element with the minimum absolute value, for each vector in a strided batch
pub unsafe fn amin_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    batch_count: i32,
    result: *mut i32,
) -> Result<()>
where
    T: AminStridedBatchedType,
{
    unsafe { T::rocblas_amin_strided_batched(handle, n, x, incx, stridex, batch_count, result) }
}

//==============================================================================
// SWAP functions
//==============================================================================

/// Interchange the contents of vectors x and y
pub unsafe fn swap<T>(handle: &Handle, n: i32, x: *mut T, incx: i32, y: *mut T, incy: i32) -> Result<()>
where
    T: SwapType,
{
    unsafe { T::rocblas_swap(handle, n, x, incx, y, incy) }
}

/// Interchange the contents of vectors x_i and y_i, for each instance in a batch
pub unsafe fn swap_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *mut T,
    incx: i32,
    y: *const *mut T,
    incy: i32,
    batch_count: i32,
) -> Result<()>
where
    T: SwapBatchedType,
{
    unsafe { T::rocblas_swap_batched(handle, n, x, incx, y, incy, batch_count) }
}

/// Interchange the contents of vectors x_i and y_i, for each instance in a strided batch
pub unsafe fn swap_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *mut T,
    incx: i32,
    stridex: i64,
    y: *mut T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
) -> Result<()>
where
    T: SwapStridedBatchedType,
{
    unsafe {
        T::rocblas_swap_strided_batched(handle, n, x, incx, stridex, y, incy, stridey, batch_count)
    }
}

//==============================================================================
// ROT / ROTG functions
//==============================================================================

/// Apply a Givens rotation matrix to a pair of vectors
///
/// The `c` parameter is always real; `s` matches the value type of `x`/`y`
/// (i.e. it is complex for the complex-valued rotations, matching `crot`/`zrot`).
pub unsafe fn rot<T>(
    handle: &Handle,
    n: i32,
    x: *mut T,
    incx: i32,
    y: *mut T,
    incy: i32,
    c: *const T::Real,
    s: *const T,
) -> Result<()>
where
    T: RotType,
{
    unsafe { T::rocblas_rot(handle, n, x, incx, y, incy, c, s) }
}

/// Apply a Givens rotation matrix to each pair of vectors in a batch
pub unsafe fn rot_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *mut T,
    incx: i32,
    y: *const *mut T,
    incy: i32,
    c: *const T::Real,
    s: *const T,
    batch_count: i32,
) -> Result<()>
where
    T: RotBatchedType,
{
    unsafe { T::rocblas_rot_batched(handle, n, x, incx, y, incy, c, s, batch_count) }
}

/// Apply a Givens rotation matrix to each pair of vectors in a strided batch
pub unsafe fn rot_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *mut T,
    incx: i32,
    stridex: i64,
    y: *mut T,
    incy: i32,
    stridey: i64,
    c: *const T::Real,
    s: *const T,
    batch_count: i32,
) -> Result<()>
where
    T: RotStridedBatchedType,
{
    unsafe {
        T::rocblas_rot_strided_batched(
            handle, n, x, incx, stridex, y, incy, stridey, c, s, batch_count,
        )
    }
}

/// Construct a Givens rotation matrix
pub unsafe fn rotg<T>(handle: &Handle, a: *mut T, b: *mut T, c: *mut T::Real, s: *mut T) -> Result<()>
where
    T: RotgType,
{
    unsafe { T::rocblas_rotg(handle, a, b, c, s) }
}

/// Construct a Givens rotation matrix for each instance in a batch
pub unsafe fn rotg_batched<T>(
    handle: &Handle,
    a: *const *mut T,
    b: *const *mut T,
    c: *const *mut T::Real,
    s: *const *mut T,
    batch_count: i32,
) -> Result<()>
where
    T: RotgBatchedType,
{
    unsafe { T::rocblas_rotg_batched(handle, a, b, c, s, batch_count) }
}

/// Construct a Givens rotation matrix for each instance in a strided batch
pub unsafe fn rotg_strided_batched<T>(
    handle: &Handle,
    a: *mut T,
    stride_a: i64,
    b: *mut T,
    stride_b: i64,
    c: *mut T::Real,
    stride_c: i64,
    s: *mut T,
    stride_s: i64,
    batch_count: i32,
) -> Result<()>
where
    T: RotgStridedBatchedType,
{
    unsafe {
        T::rocblas_rotg_strided_batched(
            handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count,
        )
    }
}

//==============================================================================
// ROTM / ROTMG functions (real types only: f32, f64)
//==============================================================================

/// Apply a modified Givens rotation matrix to a pair of vectors
pub unsafe fn rotm<T>(
    handle: &Handle,
    n: i32,
    x: *mut T,
    incx: i32,
    y: *mut T,
    incy: i32,
    param: *const T,
) -> Result<()>
where
    T: RotmType,
{
    unsafe { T::rocblas_rotm(handle, n, x, incx, y, incy, param) }
}

/// Apply a modified Givens rotation matrix to each pair of vectors in a batch
pub unsafe fn rotm_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *mut T,
    incx: i32,
    y: *const *mut T,
    incy: i32,
    param: *const *const T,
    batch_count: i32,
) -> Result<()>
where
    T: RotmBatchedType,
{
    unsafe { T::rocblas_rotm_batched(handle, n, x, incx, y, incy, param, batch_count) }
}

/// Apply a modified Givens rotation matrix to each pair of vectors in a strided batch
pub unsafe fn rotm_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *mut T,
    incx: i32,
    stridex: i64,
    y: *mut T,
    incy: i32,
    stridey: i64,
    param: *const T,
    param_stride: i64,
    batch_count: i32,
) -> Result<()>
where
    T: RotmStridedBatchedType,
{
    unsafe {
        T::rocblas_rotm_strided_batched(
            handle, n, x, incx, stridex, y, incy, stridey, param, param_stride, batch_count,
        )
    }
}

/// Construct a modified Givens rotation matrix
pub unsafe fn rotmg<T>(
    handle: &Handle,
    d1: *mut T,
    d2: *mut T,
    x1: *mut T,
    y1: *const T,
    param: *mut T,
) -> Result<()>
where
    T: RotmgType,
{
    unsafe { T::rocblas_rotmg(handle, d1, d2, x1, y1, param) }
}

/// Construct a modified Givens rotation matrix for each instance in a batch
pub unsafe fn rotmg_batched<T>(
    handle: &Handle,
    d1: *const *mut T,
    d2: *const *mut T,
    x1: *const *mut T,
    y1: *const *const T,
    param: *const *mut T,
    batch_count: i32,
) -> Result<()>
where
    T: RotmgBatchedType,
{
    unsafe { T::rocblas_rotmg_batched(handle, d1, d2, x1, y1, param, batch_count) }
}

/// Construct a modified Givens rotation matrix for each instance in a strided batch
pub unsafe fn rotmg_strided_batched<T>(
    handle: &Handle,
    d1: *mut T,
    stride_d1: i64,
    d2: *mut T,
    stride_d2: i64,
    x1: *mut T,
    stride_x1: i64,
    y1: *const T,
    stride_y1: i64,
    param: *mut T,
    stride_param: i64,
    batch_count: i32,
) -> Result<()>
where
    T: RotmgStridedBatchedType,
{
    unsafe {
        T::rocblas_rotmg_strided_batched(
            handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1, param,
            stride_param, batch_count,
        )
    }
}

//==============================================================================
// DOT - batched / strided batched (real dot, non-conjugated/conjugated)
//==============================================================================

/// Compute the dot product of each pair of vectors in a batch
pub unsafe fn dot_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    batch_count: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotBatchedType,
{
    unsafe { T::rocblas_dot_batched(handle, n, x, incx, y, incy, batch_count, result) }
}

/// Compute the dot product of each pair of vectors in a strided batch
pub unsafe fn dot_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    y: *const T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotStridedBatchedType,
{
    unsafe {
        T::rocblas_dot_strided_batched(
            handle, n, x, incx, stridex, y, incy, stridey, batch_count, result,
        )
    }
}

/// Compute the non-conjugated dot product of each pair of complex vectors in a batch
pub unsafe fn dotu_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    batch_count: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotuBatchedType,
{
    unsafe { T::rocblas_dotu_batched(handle, n, x, incx, y, incy, batch_count, result) }
}

/// Compute the non-conjugated dot product of each pair of complex vectors in a strided batch
pub unsafe fn dotu_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    y: *const T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotuStridedBatchedType,
{
    unsafe {
        T::rocblas_dotu_strided_batched(
            handle, n, x, incx, stridex, y, incy, stridey, batch_count, result,
        )
    }
}

/// Compute the conjugated dot product of each pair of complex vectors in a batch
pub unsafe fn dotc_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const *const T,
    incx: i32,
    y: *const *const T,
    incy: i32,
    batch_count: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotcBatchedType,
{
    unsafe { T::rocblas_dotc_batched(handle, n, x, incx, y, incy, batch_count, result) }
}

/// Compute the conjugated dot product of each pair of complex vectors in a strided batch
pub unsafe fn dotc_strided_batched<T>(
    handle: &Handle,
    n: i32,
    x: *const T,
    incx: i32,
    stridex: i64,
    y: *const T,
    incy: i32,
    stridey: i64,
    batch_count: i32,
    result: *mut T,
) -> Result<()>
where
    T: DotcStridedBatchedType,
{
    unsafe {
        T::rocblas_dotc_strided_batched(
            handle, n, x, incx, stridex, y, incy, stridey, batch_count, result,
        )
    }
}

//==============================================================================
// Type traits for AXPY / AMAX / AMIN / SWAP / ROTM / ROTMG / DOT batched (macro-generated)
//==============================================================================

impl_rocblas_traits!(
    AxpyType,
    AxpyTypeFn,
    {
        f32 => ffi::rocblas_saxpy,
        f64 => ffi::rocblas_daxpy,
        ffi::rocblas_float_complex => ffi::rocblas_caxpy,
        ffi::rocblas_double_complex => ffi::rocblas_zaxpy,
    },
    rocblas_axpy,
    (handle: &Handle, n: i32, alpha: &Self, x: *const Self, incx: i32, y: *mut Self, incy: i32),
    (*mut _rocblas_handle, i32, *const T, *const T, i32, *mut T, i32),
    (handle.as_raw(), n, alpha, x, incx, y, incy)
);

impl_rocblas_traits!(
    AxpyBatchedType,
    AxpyBatchedTypeFn,
    {
        f32 => ffi::rocblas_saxpy_batched,
        f64 => ffi::rocblas_daxpy_batched,
        ffi::rocblas_float_complex => ffi::rocblas_caxpy_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zaxpy_batched,
    },
    rocblas_axpy_batched,
    (handle: &Handle, n: i32, alpha: &Self, x: *const *const Self, incx: i32, y: *const *mut Self, incy: i32, batch_count: i32),
    (*mut _rocblas_handle, i32, *const T, *const *const T, i32, *const *mut T, i32, i32),
    (handle.as_raw(), n, alpha, x, incx, y, incy, batch_count)
);

impl_rocblas_traits!(
    AxpyStridedBatchedType,
    AxpyStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_saxpy_strided_batched,
        f64 => ffi::rocblas_daxpy_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_caxpy_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zaxpy_strided_batched,
    },
    rocblas_axpy_strided_batched,
    (handle: &Handle, n: i32, alpha: &Self, x: *const Self, incx: i32, stridex: i64, y: *mut Self, incy: i32, stridey: i64, batch_count: i32),
    (*mut _rocblas_handle, i32, *const T, *const T, i32, i64, *mut T, i32, i64, i32),
    (handle.as_raw(), n, alpha, x, incx, stridex, y, incy, stridey, batch_count)
);

impl_rocblas_traits!(
    AmaxType,
    AmaxTypeFn,
    {
        f32 => ffi::rocblas_isamax,
        f64 => ffi::rocblas_idamax,
        ffi::rocblas_float_complex => ffi::rocblas_icamax,
        ffi::rocblas_double_complex => ffi::rocblas_izamax,
    },
    rocblas_amax,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, result: *mut i32),
    (*mut _rocblas_handle, i32, *const T, i32, *mut i32),
    (handle.as_raw(), n, x, incx, result)
);

impl_rocblas_traits!(
    AmaxBatchedType,
    AmaxBatchedTypeFn,
    {
        f32 => ffi::rocblas_isamax_batched,
        f64 => ffi::rocblas_idamax_batched,
        ffi::rocblas_float_complex => ffi::rocblas_icamax_batched,
        ffi::rocblas_double_complex => ffi::rocblas_izamax_batched,
    },
    rocblas_amax_batched,
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, batch_count: i32, result: *mut i32),
    (*mut _rocblas_handle, i32, *const *const T, i32, i32, *mut i32),
    (handle.as_raw(), n, x, incx, batch_count, result)
);

impl_rocblas_traits!(
    AmaxStridedBatchedType,
    AmaxStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_isamax_strided_batched,
        f64 => ffi::rocblas_idamax_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_icamax_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_izamax_strided_batched,
    },
    rocblas_amax_strided_batched,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, batch_count: i32, result: *mut i32),
    (*mut _rocblas_handle, i32, *const T, i32, i64, i32, *mut i32),
    (handle.as_raw(), n, x, incx, stridex, batch_count, result)
);

impl_rocblas_traits!(
    AminType,
    AminTypeFn,
    {
        f32 => ffi::rocblas_isamin,
        f64 => ffi::rocblas_idamin,
        ffi::rocblas_float_complex => ffi::rocblas_icamin,
        ffi::rocblas_double_complex => ffi::rocblas_izamin,
    },
    rocblas_amin,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, result: *mut i32),
    (*mut _rocblas_handle, i32, *const T, i32, *mut i32),
    (handle.as_raw(), n, x, incx, result)
);

impl_rocblas_traits!(
    AminBatchedType,
    AminBatchedTypeFn,
    {
        f32 => ffi::rocblas_isamin_batched,
        f64 => ffi::rocblas_idamin_batched,
        ffi::rocblas_float_complex => ffi::rocblas_icamin_batched,
        ffi::rocblas_double_complex => ffi::rocblas_izamin_batched,
    },
    rocblas_amin_batched,
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, batch_count: i32, result: *mut i32),
    (*mut _rocblas_handle, i32, *const *const T, i32, i32, *mut i32),
    (handle.as_raw(), n, x, incx, batch_count, result)
);

impl_rocblas_traits!(
    AminStridedBatchedType,
    AminStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_isamin_strided_batched,
        f64 => ffi::rocblas_idamin_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_icamin_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_izamin_strided_batched,
    },
    rocblas_amin_strided_batched,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, batch_count: i32, result: *mut i32),
    (*mut _rocblas_handle, i32, *const T, i32, i64, i32, *mut i32),
    (handle.as_raw(), n, x, incx, stridex, batch_count, result)
);

impl_rocblas_traits!(
    SwapType,
    SwapTypeFn,
    {
        f32 => ffi::rocblas_sswap,
        f64 => ffi::rocblas_dswap,
        ffi::rocblas_float_complex => ffi::rocblas_cswap,
        ffi::rocblas_double_complex => ffi::rocblas_zswap,
    },
    rocblas_swap,
    (handle: &Handle, n: i32, x: *mut Self, incx: i32, y: *mut Self, incy: i32),
    (*mut _rocblas_handle, i32, *mut T, i32, *mut T, i32),
    (handle.as_raw(), n, x, incx, y, incy)
);

impl_rocblas_traits!(
    SwapBatchedType,
    SwapBatchedTypeFn,
    {
        f32 => ffi::rocblas_sswap_batched,
        f64 => ffi::rocblas_dswap_batched,
        ffi::rocblas_float_complex => ffi::rocblas_cswap_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zswap_batched,
    },
    rocblas_swap_batched,
    (handle: &Handle, n: i32, x: *const *mut Self, incx: i32, y: *const *mut Self, incy: i32, batch_count: i32),
    (*mut _rocblas_handle, i32, *const *mut T, i32, *const *mut T, i32, i32),
    (handle.as_raw(), n, x, incx, y, incy, batch_count)
);

impl_rocblas_traits!(
    SwapStridedBatchedType,
    SwapStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_sswap_strided_batched,
        f64 => ffi::rocblas_dswap_strided_batched,
        ffi::rocblas_float_complex => ffi::rocblas_cswap_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zswap_strided_batched,
    },
    rocblas_swap_strided_batched,
    (handle: &Handle, n: i32, x: *mut Self, incx: i32, stridex: i64, y: *mut Self, incy: i32, stridey: i64, batch_count: i32),
    (*mut _rocblas_handle, i32, *mut T, i32, i64, *mut T, i32, i64, i32),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count)
);

impl_rocblas_traits!(
    RotmType,
    RotmTypeFn,
    {
        f32 => ffi::rocblas_srotm,
        f64 => ffi::rocblas_drotm,
    },
    rocblas_rotm,
    (handle: &Handle, n: i32, x: *mut Self, incx: i32, y: *mut Self, incy: i32, param: *const Self),
    (*mut _rocblas_handle, i32, *mut T, i32, *mut T, i32, *const T),
    (handle.as_raw(), n, x, incx, y, incy, param)
);

impl_rocblas_traits!(
    RotmBatchedType,
    RotmBatchedTypeFn,
    {
        f32 => ffi::rocblas_srotm_batched,
        f64 => ffi::rocblas_drotm_batched,
    },
    rocblas_rotm_batched,
    (handle: &Handle, n: i32, x: *const *mut Self, incx: i32, y: *const *mut Self, incy: i32, param: *const *const Self, batch_count: i32),
    (*mut _rocblas_handle, i32, *const *mut T, i32, *const *mut T, i32, *const *const T, i32),
    (handle.as_raw(), n, x, incx, y, incy, param, batch_count)
);

impl_rocblas_traits!(
    RotmStridedBatchedType,
    RotmStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_srotm_strided_batched,
        f64 => ffi::rocblas_drotm_strided_batched,
    },
    rocblas_rotm_strided_batched,
    (handle: &Handle, n: i32, x: *mut Self, incx: i32, stridex: i64, y: *mut Self, incy: i32, stridey: i64, param: *const Self, param_stride: i64, batch_count: i32),
    (*mut _rocblas_handle, i32, *mut T, i32, i64, *mut T, i32, i64, *const T, i64, i32),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, param, param_stride, batch_count)
);

impl_rocblas_traits!(
    RotmgType,
    RotmgTypeFn,
    {
        f32 => ffi::rocblas_srotmg,
        f64 => ffi::rocblas_drotmg,
    },
    rocblas_rotmg,
    (handle: &Handle, d1: *mut Self, d2: *mut Self, x1: *mut Self, y1: *const Self, param: *mut Self),
    (*mut _rocblas_handle, *mut T, *mut T, *mut T, *const T, *mut T),
    (handle.as_raw(), d1, d2, x1, y1, param)
);

impl_rocblas_traits!(
    RotmgBatchedType,
    RotmgBatchedTypeFn,
    {
        f32 => ffi::rocblas_srotmg_batched,
        f64 => ffi::rocblas_drotmg_batched,
    },
    rocblas_rotmg_batched,
    (handle: &Handle, d1: *const *mut Self, d2: *const *mut Self, x1: *const *mut Self, y1: *const *const Self, param: *const *mut Self, batch_count: i32),
    (*mut _rocblas_handle, *const *mut T, *const *mut T, *const *mut T, *const *const T, *const *mut T, i32),
    (handle.as_raw(), d1, d2, x1, y1, param, batch_count)
);

impl_rocblas_traits!(
    RotmgStridedBatchedType,
    RotmgStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_srotmg_strided_batched,
        f64 => ffi::rocblas_drotmg_strided_batched,
    },
    rocblas_rotmg_strided_batched,
    (handle: &Handle, d1: *mut Self, stride_d1: i64, d2: *mut Self, stride_d2: i64, x1: *mut Self, stride_x1: i64, y1: *const Self, stride_y1: i64, param: *mut Self, stride_param: i64, batch_count: i32),
    (*mut _rocblas_handle, *mut T, i64, *mut T, i64, *mut T, i64, *const T, i64, *mut T, i64, i32),
    (handle.as_raw(), d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1, param, stride_param, batch_count)
);

impl_rocblas_traits!(
    DotBatchedType,
    DotBatchedTypeFn,
    {
        f32 => ffi::rocblas_sdot_batched,
        f64 => ffi::rocblas_ddot_batched,
        ffi::rocblas_half => ffi::rocblas_hdot_batched,
        ffi::rocblas_bfloat16 => ffi::rocblas_bfdot_batched,
    },
    rocblas_dot_batched,
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, y: *const *const Self, incy: i32, batch_count: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const *const T, i32, *const *const T, i32, i32, *mut T),
    (handle.as_raw(), n, x, incx, y, incy, batch_count, result)
);

impl_rocblas_traits!(
    DotStridedBatchedType,
    DotStridedBatchedTypeFn,
    {
        f32 => ffi::rocblas_sdot_strided_batched,
        f64 => ffi::rocblas_ddot_strided_batched,
        ffi::rocblas_half => ffi::rocblas_hdot_strided_batched,
        ffi::rocblas_bfloat16 => ffi::rocblas_bfdot_strided_batched,
    },
    rocblas_dot_strided_batched,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, y: *const Self, incy: i32, stridey: i64, batch_count: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const T, i32, i64, *const T, i32, i64, i32, *mut T),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count, result)
);

impl_rocblas_traits!(
    DotuBatchedType,
    DotuBatchedTypeFn,
    {
        ffi::rocblas_float_complex => ffi::rocblas_cdotu_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zdotu_batched,
    },
    rocblas_dotu_batched,
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, y: *const *const Self, incy: i32, batch_count: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const *const T, i32, *const *const T, i32, i32, *mut T),
    (handle.as_raw(), n, x, incx, y, incy, batch_count, result)
);

impl_rocblas_traits!(
    DotuStridedBatchedType,
    DotuStridedBatchedTypeFn,
    {
        ffi::rocblas_float_complex => ffi::rocblas_cdotu_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zdotu_strided_batched,
    },
    rocblas_dotu_strided_batched,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, y: *const Self, incy: i32, stridey: i64, batch_count: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const T, i32, i64, *const T, i32, i64, i32, *mut T),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count, result)
);

impl_rocblas_traits!(
    DotcBatchedType,
    DotcBatchedTypeFn,
    {
        ffi::rocblas_float_complex => ffi::rocblas_cdotc_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zdotc_batched,
    },
    rocblas_dotc_batched,
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, y: *const *const Self, incy: i32, batch_count: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const *const T, i32, *const *const T, i32, i32, *mut T),
    (handle.as_raw(), n, x, incx, y, incy, batch_count, result)
);

impl_rocblas_traits!(
    DotcStridedBatchedType,
    DotcStridedBatchedTypeFn,
    {
        ffi::rocblas_float_complex => ffi::rocblas_cdotc_strided_batched,
        ffi::rocblas_double_complex => ffi::rocblas_zdotc_strided_batched,
    },
    rocblas_dotc_strided_batched,
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, y: *const Self, incy: i32, stridey: i64, batch_count: i32, result: *mut Self),
    (*mut _rocblas_handle, i32, *const T, i32, i64, *const T, i32, i64, i32, *mut T),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, batch_count, result)
);

//==============================================================================
// Type traits for NRM2 / ASUM (manual impls: real result type differs from T)
//==============================================================================

macro_rules! impl_real_result_trait {
    ($trait_name:ident, $method:ident, { $($t:ty, $r:ty => $func:path),* $(,)? }, $args:tt, $call_args:tt) => {
        pub trait $trait_name {
            type Result;
            unsafe fn $method $args -> Result<()>;
        }
        $(
            impl $trait_name for $t {
                type Result = $r;

                unsafe fn $method $args -> Result<()> {
                    let status = unsafe { $func $call_args };
                    if status != ffi::rocblas_status__rocblas_status_success {
                        return Err(Error::new(status));
                    }
                    Ok(())
                }
            }
        )*
    };
}

impl_real_result_trait!(
    Nrm2Type,
    rocblas_nrm2,
    {
        f32, f32 => ffi::rocblas_snrm2,
        f64, f64 => ffi::rocblas_dnrm2,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_scnrm2,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_dznrm2,
    },
    (handle: &Handle, n: i32, x: *const Self, incx: i32, result: *mut Self::Result),
    (handle.as_raw(), n, x, incx, result)
);

impl_real_result_trait!(
    Nrm2BatchedType,
    rocblas_nrm2_batched,
    {
        f32, f32 => ffi::rocblas_snrm2_batched,
        f64, f64 => ffi::rocblas_dnrm2_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_scnrm2_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_dznrm2_batched,
    },
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, batch_count: i32, result: *mut Self::Result),
    (handle.as_raw(), n, x, incx, batch_count, result)
);

impl_real_result_trait!(
    Nrm2StridedBatchedType,
    rocblas_nrm2_strided_batched,
    {
        f32, f32 => ffi::rocblas_snrm2_strided_batched,
        f64, f64 => ffi::rocblas_dnrm2_strided_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_scnrm2_strided_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_dznrm2_strided_batched,
    },
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, batch_count: i32, result: *mut Self::Result),
    (handle.as_raw(), n, x, incx, stridex, batch_count, result)
);

impl_real_result_trait!(
    AsumType,
    rocblas_asum,
    {
        f32, f32 => ffi::rocblas_sasum,
        f64, f64 => ffi::rocblas_dasum,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_scasum,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_dzasum,
    },
    (handle: &Handle, n: i32, x: *const Self, incx: i32, result: *mut Self::Result),
    (handle.as_raw(), n, x, incx, result)
);

impl_real_result_trait!(
    AsumBatchedType,
    rocblas_asum_batched,
    {
        f32, f32 => ffi::rocblas_sasum_batched,
        f64, f64 => ffi::rocblas_dasum_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_scasum_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_dzasum_batched,
    },
    (handle: &Handle, n: i32, x: *const *const Self, incx: i32, batch_count: i32, result: *mut Self::Result),
    (handle.as_raw(), n, x, incx, batch_count, result)
);

impl_real_result_trait!(
    AsumStridedBatchedType,
    rocblas_asum_strided_batched,
    {
        f32, f32 => ffi::rocblas_sasum_strided_batched,
        f64, f64 => ffi::rocblas_dasum_strided_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_scasum_strided_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_dzasum_strided_batched,
    },
    (handle: &Handle, n: i32, x: *const Self, incx: i32, stridex: i64, batch_count: i32, result: *mut Self::Result),
    (handle.as_raw(), n, x, incx, stridex, batch_count, result)
);

//==============================================================================
// Type traits for ROT / ROTG (manual impls: real component type differs from T)
//==============================================================================

macro_rules! impl_rot_trait {
    ($trait_name:ident, $method:ident, { $($t:ty, $r:ty => $func:path),* $(,)? }, $args:tt, $call_args:tt) => {
        pub trait $trait_name {
            type Real;
            unsafe fn $method $args -> Result<()>;
        }
        $(
            impl $trait_name for $t {
                type Real = $r;

                unsafe fn $method $args -> Result<()> {
                    let status = unsafe { $func $call_args };
                    if status != ffi::rocblas_status__rocblas_status_success {
                        return Err(Error::new(status));
                    }
                    Ok(())
                }
            }
        )*
    };
}

impl_rot_trait!(
    RotType,
    rocblas_rot,
    {
        f32, f32 => ffi::rocblas_srot,
        f64, f64 => ffi::rocblas_drot,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_crot,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_zrot,
    },
    (handle: &Handle, n: i32, x: *mut Self, incx: i32, y: *mut Self, incy: i32, c: *const Self::Real, s: *const Self),
    (handle.as_raw(), n, x, incx, y, incy, c, s)
);

impl_rot_trait!(
    RotBatchedType,
    rocblas_rot_batched,
    {
        f32, f32 => ffi::rocblas_srot_batched,
        f64, f64 => ffi::rocblas_drot_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_crot_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_zrot_batched,
    },
    (handle: &Handle, n: i32, x: *const *mut Self, incx: i32, y: *const *mut Self, incy: i32, c: *const Self::Real, s: *const Self, batch_count: i32),
    (handle.as_raw(), n, x, incx, y, incy, c, s, batch_count)
);

impl_rot_trait!(
    RotStridedBatchedType,
    rocblas_rot_strided_batched,
    {
        f32, f32 => ffi::rocblas_srot_strided_batched,
        f64, f64 => ffi::rocblas_drot_strided_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_crot_strided_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_zrot_strided_batched,
    },
    (handle: &Handle, n: i32, x: *mut Self, incx: i32, stridex: i64, y: *mut Self, incy: i32, stridey: i64, c: *const Self::Real, s: *const Self, batch_count: i32),
    (handle.as_raw(), n, x, incx, stridex, y, incy, stridey, c, s, batch_count)
);

impl_rot_trait!(
    RotgType,
    rocblas_rotg,
    {
        f32, f32 => ffi::rocblas_srotg,
        f64, f64 => ffi::rocblas_drotg,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_crotg,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_zrotg,
    },
    (handle: &Handle, a: *mut Self, b: *mut Self, c: *mut Self::Real, s: *mut Self),
    (handle.as_raw(), a, b, c, s)
);

impl_rot_trait!(
    RotgBatchedType,
    rocblas_rotg_batched,
    {
        f32, f32 => ffi::rocblas_srotg_batched,
        f64, f64 => ffi::rocblas_drotg_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_crotg_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_zrotg_batched,
    },
    (handle: &Handle, a: *const *mut Self, b: *const *mut Self, c: *const *mut Self::Real, s: *const *mut Self, batch_count: i32),
    (handle.as_raw(), a, b, c, s, batch_count)
);

impl_rot_trait!(
    RotgStridedBatchedType,
    rocblas_rotg_strided_batched,
    {
        f32, f32 => ffi::rocblas_srotg_strided_batched,
        f64, f64 => ffi::rocblas_drotg_strided_batched,
        ffi::rocblas_float_complex, f32 => ffi::rocblas_crotg_strided_batched,
        ffi::rocblas_double_complex, f64 => ffi::rocblas_zrotg_strided_batched,
    },
    (handle: &Handle, a: *mut Self, stride_a: i64, b: *mut Self, stride_b: i64, c: *mut Self::Real, stride_c: i64, s: *mut Self, stride_s: i64, batch_count: i32),
    (handle.as_raw(), a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
);

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(data: &[f32]) -> DeviceMemory<f32> {
        let mut m = DeviceMemory::<f32>::new(data.len()).unwrap();
        m.copy_from_host(data).unwrap();
        m
    }

    #[test]
    fn test_nrm2() {
        let handle = Handle::new().unwrap();
        let x = dev(&[3.0, 4.0]);
        let mut result = 0.0f32;
        unsafe {
            nrm2(&handle, 2, x.as_ptr().cast::<f32>(), 1, &mut result).unwrap();
        }
        assert!((result - 5.0).abs() < 1e-4, "nrm2 = {result}");
    }

    #[test]
    fn test_asum() {
        let handle = Handle::new().unwrap();
        let x = dev(&[1.0, -2.0, 3.0]);
        let mut result = 0.0f32;
        unsafe {
            asum(&handle, 3, x.as_ptr().cast::<f32>(), 1, &mut result).unwrap();
        }
        assert!((result - 6.0).abs() < 1e-4, "asum = {result}");
    }

    #[test]
    fn test_amax() {
        let handle = Handle::new().unwrap();
        let x = dev(&[1.0, -5.0, 3.0]);
        let mut idx = 0i32;
        unsafe {
            amax(&handle, 3, x.as_ptr().cast::<f32>(), 1, &mut idx).unwrap();
        }
        // rocBLAS returns a 1-based index; |-5| is the largest magnitude.
        assert_eq!(idx, 2, "amax index = {idx}");
    }

    #[test]
    fn test_amin() {
        let handle = Handle::new().unwrap();
        let x = dev(&[1.0, -5.0, 3.0]);
        let mut idx = 0i32;
        unsafe {
            amin(&handle, 3, x.as_ptr().cast::<f32>(), 1, &mut idx).unwrap();
        }
        // 1-based index; |1| is the smallest magnitude.
        assert_eq!(idx, 1, "amin index = {idx}");
    }
}
