// src/rocblas/safe.rs
//
// A thin, memory-safe convenience layer over the raw `unsafe` rocBLAS wrappers.
//
// The functions in the parent module (`level1`, `level2`, `level3`) mirror the C
// rocBLAS API 1:1 and are therefore `unsafe`, operating on raw device pointers.
// That layer is complete and maximally flexible, but it pushes all pointer/size
// correctness onto the caller.
//
// This module offers an ergonomic alternative for the most common operations. It
// works directly on the crate's [`DeviceMemory<T>`] wrapper, derives the vector
// length and leading dimensions from the allocations where possible, validates
// sizes up front, and exposes ordinary safe `fn`s. It does not attempt to cover
// every batched/strided variant — for those, drop down to the raw layer.

use crate::hip::DeviceMemory;
use crate::rocblas::error::{Error, Result};
use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::types::{Fill, Operation};
use crate::rocblas::{level1, level2, level3};

/// Return `rocblas_status_invalid_size` as an [`Error`].
#[inline]
fn invalid_size() -> Error {
    Error::new(ffi::rocblas_status__rocblas_status_invalid_size)
}

/// Number of elements addressed by a strided vector of logical length `n`.
///
/// For an increment `incx`, element `i` lives at offset `i * incx`, so the
/// allocation must hold at least `1 + (n - 1) * |incx|` elements.
#[inline]
fn strided_len(n: i32, inc: i32) -> Option<usize> {
    if n <= 0 || inc == 0 {
        return None;
    }
    let n = n as i64;
    let inc = inc.unsigned_abs() as i64;
    Some((1 + (n - 1) * inc) as usize)
}

/// Validate that `mem` can hold a strided vector of logical length `n` with
/// increment `inc`.
#[inline]
fn check_vector<T>(mem: &DeviceMemory<T>, n: i32, inc: i32) -> Result<()> {
    match strided_len(n, inc) {
        Some(required) if mem.count() >= required => Ok(()),
        _ => Err(invalid_size()),
    }
}

//==============================================================================
// Level 1
//==============================================================================

/// `x := alpha * x`
///
/// Scales the elements of `x` in place. `incx` may be greater than 1 to operate
/// on strided data. Returns an error if `x` is too small for the requested
/// length/stride.
pub fn scal<T>(handle: &Handle, alpha: &T, x: &mut DeviceMemory<T>, incx: i32) -> Result<()>
where
    T: level1::ScalType,
{
    let n = vector_len(x.count(), incx)?;
    check_vector(x, n, incx)?;
    // `level1::scal` is safe and takes `&DeviceMemory<T>`.
    level1::scal(handle, n, alpha, x, incx)
}

/// `y := alpha * x + y`
///
/// Both vectors must be large enough for `n` elements at their respective
/// increments, where `n` is derived from `x`.
pub fn axpy<T>(
    handle: &Handle,
    alpha: &T,
    x: &DeviceMemory<T>,
    incx: i32,
    y: &mut DeviceMemory<T>,
    incy: i32,
) -> Result<()>
where
    T: level1::AxpyType,
{
    let n = vector_len(x.count(), incx)?;
    check_vector(x, n, incx)?;
    check_vector(y, n, incy)?;
    unsafe {
        level1::axpy(
            handle,
            n,
            alpha,
            x.as_ptr().cast::<T>(),
            incx,
            y.as_ptr().cast::<T>(),
            incy,
        )
    }
}

/// `y := x`
///
/// Copies `n` elements from `x` into `y`, where `n` is derived from `x`.
pub fn copy<T>(
    handle: &Handle,
    x: &DeviceMemory<T>,
    incx: i32,
    y: &mut DeviceMemory<T>,
    incy: i32,
) -> Result<()>
where
    T: level1::CopyType,
{
    let n = vector_len(x.count(), incx)?;
    check_vector(x, n, incx)?;
    check_vector(y, n, incy)?;
    unsafe {
        level1::copy(
            handle,
            n,
            x.as_ptr().cast::<T>(),
            incx,
            y.as_ptr().cast::<T>(),
            incy,
        )
    }
}

/// `x, y := y, x`
///
/// Swaps `n` elements between `x` and `y`, where `n` is derived from `x`.
pub fn swap<T>(
    handle: &Handle,
    x: &mut DeviceMemory<T>,
    incx: i32,
    y: &mut DeviceMemory<T>,
    incy: i32,
) -> Result<()>
where
    T: level1::SwapType,
{
    let n = vector_len(x.count(), incx)?;
    check_vector(x, n, incx)?;
    check_vector(y, n, incy)?;
    unsafe {
        level1::swap(
            handle,
            n,
            x.as_ptr().cast::<T>(),
            incx,
            y.as_ptr().cast::<T>(),
            incy,
        )
    }
}

/// Dot product `result := x^T * y`.
///
/// The result is written into the host-side `out`.
pub fn dot<T>(
    handle: &Handle,
    x: &DeviceMemory<T>,
    incx: i32,
    y: &DeviceMemory<T>,
    incy: i32,
    out: &mut T,
) -> Result<()>
where
    T: level1::DotType,
{
    let n = vector_len(x.count(), incx)?;
    check_vector(x, n, incx)?;
    check_vector(y, n, incy)?;
    unsafe {
        level1::dot(
            handle,
            n,
            x.as_ptr().cast::<T>(),
            incx,
            y.as_ptr().cast::<T>(),
            incy,
            out as *mut T,
        )
    }
}

//==============================================================================
// Level 2
//==============================================================================

/// General matrix-vector multiply `y := alpha * op(A) * x + beta * y`.
///
/// `A` is stored column-major with leading dimension `lda` and logical shape
/// `rows x cols`. The vector lengths are derived from `op(A)`:
/// `op(A)` is `rows x cols` when `trans == None`, else `cols x rows`.
#[allow(clippy::too_many_arguments)]
pub fn gemv<T>(
    handle: &Handle,
    trans: Operation,
    rows: i32,
    cols: i32,
    alpha: &T,
    a: &DeviceMemory<T>,
    lda: i32,
    x: &DeviceMemory<T>,
    incx: i32,
    beta: &T,
    y: &mut DeviceMemory<T>,
    incy: i32,
) -> Result<()>
where
    T: level2::GemvType,
{
    if rows <= 0 || cols <= 0 || lda < rows {
        return Err(invalid_size());
    }
    check_matrix(a, lda, cols)?;
    // Length of x (columns of op(A)) and y (rows of op(A)).
    let (len_x, len_y) = match trans {
        Operation::None => (cols, rows),
        _ => (rows, cols),
    };
    check_vector(x, len_x, incx)?;
    check_vector(y, len_y, incy)?;
    unsafe {
        level2::gemv(
            handle,
            trans,
            rows,
            cols,
            alpha,
            a.as_ptr().cast::<T>(),
            lda,
            x.as_ptr().cast::<T>(),
            incx,
            beta,
            y.as_ptr().cast::<T>(),
            incy,
        )
    }
}

//==============================================================================
// Level 3
//==============================================================================

/// General matrix-matrix multiply `C := alpha * op(A) * op(B) + beta * C`.
///
/// All matrices are column-major. `C` is `m x n`, `op(A)` is `m x k`, and
/// `op(B)` is `k x n`. Leading dimensions are validated against the (untransposed)
/// physical shapes.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T>(
    handle: &Handle,
    trans_a: Operation,
    trans_b: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    a: &DeviceMemory<T>,
    lda: i32,
    b: &DeviceMemory<T>,
    ldb: i32,
    beta: &T,
    c: &mut DeviceMemory<T>,
    ldc: i32,
) -> Result<()>
where
    T: level3::GemmType,
{
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(invalid_size());
    }
    // Physical (row, col) extents of each operand given the transpose flags.
    let (a_rows, a_cols) = match trans_a {
        Operation::None => (m, k),
        _ => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        Operation::None => (k, n),
        _ => (n, k),
    };
    if lda < a_rows || ldb < b_rows || ldc < m {
        return Err(invalid_size());
    }
    check_matrix(a, lda, a_cols)?;
    check_matrix(b, ldb, b_cols)?;
    check_matrix(c, ldc, n)?;
    unsafe {
        level3::gemm(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a.as_ptr().cast::<T>(),
            lda,
            b.as_ptr().cast::<T>(),
            ldb,
            beta,
            c.as_ptr().cast::<T>(),
            ldc,
        )
    }
}

/// Symmetric rank-k update `C := alpha * op(A) * op(A)^T + beta * C`.
///
/// `C` is an `n x n` symmetric matrix; only the triangle selected by `uplo` is
/// referenced/updated. `op(A)` is `n x k`.
#[allow(clippy::too_many_arguments)]
pub fn syrk<T>(
    handle: &Handle,
    uplo: Fill,
    trans_a: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    a: &DeviceMemory<T>,
    lda: i32,
    beta: &T,
    c: &mut DeviceMemory<T>,
    ldc: i32,
) -> Result<()>
where
    T: level3::SyrkType,
{
    if n <= 0 || k <= 0 || ldc < n {
        return Err(invalid_size());
    }
    let a_rows = match trans_a {
        Operation::None => n,
        _ => k,
    };
    let a_cols = match trans_a {
        Operation::None => k,
        _ => n,
    };
    if lda < a_rows {
        return Err(invalid_size());
    }
    check_matrix(a, lda, a_cols)?;
    check_matrix(c, ldc, n)?;
    unsafe {
        level3::syrk(
            handle,
            uplo,
            trans_a,
            n,
            k,
            alpha,
            a.as_ptr().cast::<T>(),
            lda,
            beta,
            c.as_ptr().cast::<T>(),
            ldc,
        )
    }
}

//==============================================================================
// Internal helpers
//==============================================================================

/// Derive the logical vector length that fits in `count` elements at the given
/// increment. For `inc == 1` this is simply `count`; for larger increments it is
/// `1 + (count - 1) / inc`.
#[inline]
fn vector_len(count: usize, inc: i32) -> Result<i32> {
    if inc == 0 {
        return Err(invalid_size());
    }
    let inc = inc.unsigned_abs() as usize;
    if count == 0 {
        return Ok(0);
    }
    let n = 1 + (count - 1) / inc;
    i32::try_from(n).map_err(|_| invalid_size())
}

/// Validate that a column-major matrix with leading dimension `lda` and `cols`
/// columns fits within `mem`.
#[inline]
fn check_matrix<T>(mem: &DeviceMemory<T>, lda: i32, cols: i32) -> Result<()> {
    if lda <= 0 || cols <= 0 {
        return Err(invalid_size());
    }
    let required = (lda as i64) * (cols as i64);
    match usize::try_from(required) {
        Ok(required) if mem.count() >= required => Ok(()),
        _ => Err(invalid_size()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upload a host slice to a fresh device allocation.
    fn dev(data: &[f32]) -> DeviceMemory<f32> {
        let mut m = DeviceMemory::<f32>::new(data.len()).unwrap();
        m.copy_from_host(data).unwrap();
        m
    }

    /// Download `n` elements from device memory back to the host.
    fn host(m: &DeviceMemory<f32>, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        m.copy_to_host(&mut v).unwrap();
        v
    }

    fn approx(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-4, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn test_scal() {
        let handle = Handle::new().unwrap();
        let mut x = dev(&[1.0, 2.0, 3.0, 4.0]);
        scal(&handle, &2.0, &mut x, 1).unwrap();
        approx(&host(&x, 4), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_axpy() {
        let handle = Handle::new().unwrap();
        let x = dev(&[1.0, 2.0, 3.0]);
        let mut y = dev(&[10.0, 20.0, 30.0]);
        axpy(&handle, &2.0, &x, 1, &mut y, 1).unwrap();
        approx(&host(&y, 3), &[12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_copy() {
        let handle = Handle::new().unwrap();
        let x = dev(&[7.0, 8.0, 9.0]);
        let mut y = dev(&[0.0, 0.0, 0.0]);
        copy(&handle, &x, 1, &mut y, 1).unwrap();
        approx(&host(&y, 3), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_swap() {
        let handle = Handle::new().unwrap();
        let mut x = dev(&[1.0, 2.0]);
        let mut y = dev(&[3.0, 4.0]);
        swap(&handle, &mut x, 1, &mut y, 1).unwrap();
        approx(&host(&x, 2), &[3.0, 4.0]);
        approx(&host(&y, 2), &[1.0, 2.0]);
    }

    #[test]
    fn test_dot() {
        let handle = Handle::new().unwrap();
        let x = dev(&[1.0, 2.0, 3.0]);
        let y = dev(&[4.0, 5.0, 6.0]);
        let mut out = 0.0f32;
        dot(&handle, &x, 1, &y, 1, &mut out).unwrap();
        assert!((out - 32.0).abs() < 1e-4, "dot = {out}");
    }

    #[test]
    fn test_gemv() {
        let handle = Handle::new().unwrap();
        // A = [[1, 2], [3, 4]] stored column-major.
        let a = dev(&[1.0, 3.0, 2.0, 4.0]);
        let x = dev(&[1.0, 1.0]);
        let mut y = dev(&[0.0, 0.0]);
        gemv(
            &handle,
            Operation::None,
            2,
            2,
            &1.0,
            &a,
            2,
            &x,
            1,
            &0.0,
            &mut y,
            1,
        )
        .unwrap();
        approx(&host(&y, 2), &[3.0, 7.0]);
    }

    #[test]
    fn test_gemm() {
        let handle = Handle::new().unwrap();
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] (column-major).
        let a = dev(&[1.0, 3.0, 2.0, 4.0]);
        let b = dev(&[5.0, 7.0, 6.0, 8.0]);
        let mut c = dev(&[0.0, 0.0, 0.0, 0.0]);
        gemm(
            &handle,
            Operation::None,
            Operation::None,
            2,
            2,
            2,
            &1.0,
            &a,
            2,
            &b,
            2,
            &0.0,
            &mut c,
            2,
        )
        .unwrap();
        // C = A*B = [[19, 22], [43, 50]] column-major.
        approx(&host(&c, 4), &[19.0, 43.0, 22.0, 50.0]);
    }

    #[test]
    fn test_syrk() {
        let handle = Handle::new().unwrap();
        // A = [[1, 2], [3, 4]] column-major; C := A * A^T, lower triangle.
        let a = dev(&[1.0, 3.0, 2.0, 4.0]);
        let mut c = dev(&[0.0, 0.0, 0.0, 0.0]);
        syrk(
            &handle,
            Fill::Lower,
            Operation::None,
            2,
            2,
            &1.0,
            &a,
            2,
            &0.0,
            &mut c,
            2,
        )
        .unwrap();
        // A*A^T = [[5, 11], [11, 25]]; only the lower triangle is written.
        let out = host(&c, 4);
        assert!((out[0] - 5.0).abs() < 1e-4, "c[0,0] = {}", out[0]);
        assert!((out[1] - 11.0).abs() < 1e-4, "c[1,0] = {}", out[1]);
        assert!((out[3] - 25.0).abs() < 1e-4, "c[1,1] = {}", out[3]);
    }

    #[test]
    fn test_size_validation_rejects_small_output() {
        let handle = Handle::new().unwrap();
        let a = dev(&[1.0, 3.0, 2.0, 4.0]);
        let b = dev(&[5.0, 7.0, 6.0, 8.0]);
        // C is too small (needs 4 elements for a 2x2 result).
        let mut c = dev(&[0.0]);
        let err = gemm(
            &handle,
            Operation::None,
            Operation::None,
            2,
            2,
            2,
            &1.0,
            &a,
            2,
            &b,
            2,
            &0.0,
            &mut c,
            2,
        );
        assert!(err.is_err(), "expected invalid_size error for undersized C");
    }
}
