// src/rocblas/level3.rs

use crate::rocblas::error::{Error, Result};
use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::types::{DataType, Operation};
use crate::rocblas::utils::GemmAlgo;

use super::types::{Diagonal, Fill, Side};

//==============================================================================
// GEMM functions - General Matrix-Matrix Multiplication
//==============================================================================

/// Matrix-matrix multiplication
///
/// Computes one of the following matrix-matrix operations:
///
/// C := alpha * A * B + beta * C
/// C := alpha * A^T * B + beta * C
/// C := alpha * A * B^T + beta * C
/// C := alpha * A^T * B^T + beta * C
///
/// where alpha and beta are scalars, and A, B, C are matrices.
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A) and C
/// * `n` - Number of columns of matrix op(B) and C
/// * `k` - Number of columns of matrix op(A) and rows of op(B)
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `B` - Buffer storing matrix B
/// * `ldb` - Leading dimension of matrix B
/// * `beta` - Scalar beta
/// * `C` - Buffer storing matrix C
/// * `ldc` - Leading dimension of matrix C
pub unsafe fn gemm<T>(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    beta: &T,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: GemmType,
{
    unsafe {
        T::rocblas_gemm(
            handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
        )
    }
}

/// Batched matrix-matrix multiplication
///
/// Computes one of the following batched matrix-matrix operations:
///
/// C_i := alpha * op(A_i) * op(B_i) + beta * C_i
///
/// where (A_i, B_i, C_i) is the i-th instance of the batch.
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A_i) and C_i
/// * `n` - Number of columns of matrix op(B_i) and C_i
/// * `k` - Number of columns of matrix op(A_i) and rows of op(B_i)
/// * `alpha` - Scalar alpha
/// * `A` - Array of pointers to matrices A_i
/// * `lda` - Leading dimension of matrices A_i
/// * `B` - Array of pointers to matrices B_i
/// * `ldb` - Leading dimension of matrices B_i
/// * `beta` - Scalar beta
/// * `C` - Array of pointers to matrices C_i
/// * `ldc` - Leading dimension of matrices C_i
/// * `batch_count` - Number of instances in the batch
pub unsafe fn gemm_batched<T>(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    B: *const *const T,
    ldb: i32,
    beta: &T,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: GemmBatchedType,
{
    unsafe {
        T::rocblas_gemm_batched(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            beta,
            C,
            ldc,
            batch_count,
        )
    }
}

/// Strided batched matrix-matrix multiplication
///
/// Computes one of the following strided batched matrix-matrix operations:
///
/// C_i := alpha * op(A_i) * op(B_i) + beta * C_i
///
/// where (A_i, B_i, C_i) is the i-th instance of the batch.
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A_i) and C_i
/// * `n` - Number of columns of matrix op(B_i) and C_i
/// * `k` - Number of columns of matrix op(A_i) and rows of op(B_i)
/// * `alpha` - Scalar alpha
/// * `A` - Pointer to the first matrix A_1
/// * `lda` - Leading dimension of matrices A_i
/// * `stride_A` - Stride from start of one matrix (A_i) to the next (A_i+1)
/// * `B` - Pointer to the first matrix B_1
/// * `ldb` - Leading dimension of matrices B_i
/// * `stride_B` - Stride from start of one matrix (B_i) to the next (B_i+1)
/// * `beta` - Scalar beta
/// * `C` - Pointer to the first matrix C_1
/// * `ldc` - Leading dimension of matrices C_i
/// * `stride_C` - Stride from start of one matrix (C_i) to the next (C_i+1)
/// * `batch_count` - Number of instances in the batch
pub unsafe fn gemm_strided_batched<T>(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    B: *const T,
    ldb: i32,
    stride_B: i64,
    beta: &T,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: GemmStridedBatchedType,
{
    unsafe {
        T::rocblas_gemm_strided_batched(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            A,
            lda,
            stride_A,
            B,
            ldb,
            stride_B,
            beta,
            C,
            ldc,
            stride_C,
            batch_count,
        )
    }
}

/// General matrix-matrix multiplication with extended precision
///
/// Computes the general matrix-matrix product with extended precision
/// where the data types of matrices can be different.
///
/// C := alpha * op(A) * op(B) + beta * C
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `transa` - Operation op(A) that is non-or (conjugate) transpose
/// * `transb` - Operation op(B) that is non-or (conjugate) transpose
/// * `m` - Number of rows of matrix op(A) and C
/// * `n` - Number of columns of matrix op(B) and C
/// * `k` - Number of columns of matrix op(A) and rows of op(B)
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `a_type` - Data type of matrix A
/// * `lda` - Leading dimension of matrix A
/// * `B` - Buffer storing matrix B
/// * `b_type` - Data type of matrix B
/// * `ldb` - Leading dimension of matrix B
/// * `beta` - Scalar beta
/// * `C` - Buffer storing matrix C
/// * `c_type` - Data type of matrix C
/// * `ldc` - Leading dimension of matrix C
/// * `compute_type` - Computation type
/// * `algo` - GEMM algorithm
pub unsafe fn gemm_ex(
    handle: &Handle,
    transa: Operation,
    transb: Operation,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const std::ffi::c_void,
    A: *const std::ffi::c_void,
    a_type: DataType,
    lda: i32,
    B: *const std::ffi::c_void,
    b_type: DataType,
    ldb: i32,
    beta: *const std::ffi::c_void,
    C: *mut std::ffi::c_void,
    c_type: DataType,
    ldc: i32,
    compute_type: DataType,
    algo: GemmAlgo,
) -> Result<()> {
    let status = unsafe {
        ffi::rocblas_gemm_ex(
            handle.as_raw(),
            transa.into(),
            transb.into(),
            m,
            n,
            k,
            alpha,
            A,
            a_type.into(),
            lda,
            B,
            b_type.into(),
            ldb,
            beta,
            C,
            c_type.into(),
            ldc,
            C, // Output matrix same as C
            c_type.into(),
            ldc,
            compute_type.into(),
            algo.into(),
            0, // Solution index (0 means auto)
            0, // Flags
        )
    };

    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    Ok(())
}

//==============================================================================
// Type traits for implementation
//==============================================================================

/// Trait for types that can be used with gemm
pub trait GemmType {
    unsafe fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl GemmType for f32 {
    unsafe fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmType for f64 {
    unsafe fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmType for ffi::rocblas_float_complex {
    unsafe fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmType for ffi::rocblas_double_complex {
    unsafe fn rocblas_gemm(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemm(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gemm_batched
pub trait GemmBatchedType {
    unsafe fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl GemmBatchedType for f32 {
    unsafe fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmBatchedType for f64 {
    unsafe fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_gemm_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemm_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with gemm_strided_batched
pub trait GemmStridedBatchedType {
    unsafe fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl GemmStridedBatchedType for f32 {
    unsafe fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmStridedBatchedType for f64 {
    unsafe fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmStridedBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl GemmStridedBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_gemm_strided_batched(
        handle: &Handle,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zgemm_strided_batched(
                handle.as_raw(),
                transa.into(),
                transb.into(),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait definitions for HEMM operations
pub trait HemmType {
    unsafe fn rocblas_hemm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl HemmType for ffi::rocblas_float_complex {
    unsafe fn rocblas_hemm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chemm(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HemmType for ffi::rocblas_double_complex {
    unsafe fn rocblas_hemm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhemm(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for HERK operations
pub trait HerkType {
    type ScalarType;

    unsafe fn rocblas_herk(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl HerkType for ffi::rocblas_float_complex {
    type ScalarType = f32;

    unsafe fn rocblas_herk(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherk(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HerkType for ffi::rocblas_double_complex {
    type ScalarType = f64;

    unsafe fn rocblas_herk(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherk(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait definitions for SPR operations (packed symmetric rank-1 update)
pub trait SprType {
    unsafe fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()>;
}

impl SprType for f32 {
    unsafe fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_sspr(handle.as_raw(), uplo.into(), n, alpha, x, incx, AP) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprType for f64 {
    unsafe fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_dspr(handle.as_raw(), uplo.into(), n, alpha, x, incx, AP) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// There are also complex versions in the bindings
impl SprType for ffi::rocblas_float_complex {
    unsafe fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_cspr(handle.as_raw(), uplo.into(), n, alpha, x, incx, AP) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SprType for ffi::rocblas_double_complex {
    unsafe fn rocblas_spr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_zspr(handle.as_raw(), uplo.into(), n, alpha, x, incx, AP) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for SPR2 operations (packed symmetric rank-2 update)
pub trait Spr2Type {
    unsafe fn rocblas_spr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        AP: *mut Self,
    ) -> Result<()>;
}

impl Spr2Type for f32 {
    unsafe fn rocblas_spr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_sspr2(handle.as_raw(), uplo.into(), n, alpha, x, incx, y, incy, AP)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Spr2Type for f64 {
    unsafe fn rocblas_spr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        AP: *mut Self,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dspr2(handle.as_raw(), uplo.into(), n, alpha, x, incx, y, incy, AP)
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for SYR operations (symmetric rank-1 update)
pub trait SyrType {
    unsafe fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()>;
}

impl SyrType for f32 {
    unsafe fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_ssyr(handle.as_raw(), uplo.into(), n, alpha, x, incx, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SyrType for f64 {
    unsafe fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_dsyr(handle.as_raw(), uplo.into(), n, alpha, x, incx, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Trait for SYR2 operations (symmetric rank-2 update)
pub trait Syr2Type {
    unsafe fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()>;
}

impl Syr2Type for f32 {
    unsafe fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2Type for f64 {
    unsafe fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_dsyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Implementations for complex versions (CSYR, ZSYR, CSYR2, ZSYR2)
impl SyrType for ffi::rocblas_float_complex {
    unsafe fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_csyr(handle.as_raw(), uplo.into(), n, alpha, x, incx, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl SyrType for ffi::rocblas_double_complex {
    unsafe fn rocblas_syr(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status =
            unsafe { ffi::rocblas_zsyr(handle.as_raw(), uplo.into(), n, alpha, x, incx, A, lda) };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2Type for ffi::rocblas_float_complex {
    unsafe fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_csyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl Syr2Type for ffi::rocblas_double_complex {
    unsafe fn rocblas_syr2(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        y: *const Self,
        incy: i32,
        A: *mut Self,
        lda: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zsyr2(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                y,
                incy,
                A,
                lda,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Batched and strided batched implementations for SYR and SYR2

pub trait SyrBatchedType {
    unsafe fn rocblas_syr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl SyrBatchedType for f32 {
    unsafe fn rocblas_syr_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const *const Self,
        incx: i32,
        A: *const *mut Self,
        lda: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                A,
                lda,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Similar implementations for other data types and strided batched versions

pub trait SyrStridedBatchedType {
    unsafe fn rocblas_syr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl SyrStridedBatchedType for f32 {
    unsafe fn rocblas_syr_strided_batched(
        handle: &Handle,
        uplo: Fill,
        n: i32,
        alpha: &Self,
        x: *const Self,
        incx: i32,
        stride_x: i64,
        A: *mut Self,
        lda: i32,
        stride_A: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_ssyr_strided_batched(
                handle.as_raw(),
                uplo.into(),
                n,
                alpha,
                x,
                incx,
                stride_x,
                A,
                lda,
                stride_A,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Additional implementations for batched versions of HEMM and HERK

pub trait HemmBatchedType {
    unsafe fn rocblas_hemm_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl HemmBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_hemm_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chemm_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HemmBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_hemm_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhemm_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait HemmStridedBatchedType {
    unsafe fn rocblas_hemm_strided_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HemmStridedBatchedType for ffi::rocblas_float_complex {
    unsafe fn rocblas_hemm_strided_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_chemm_strided_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HemmStridedBatchedType for ffi::rocblas_double_complex {
    unsafe fn rocblas_hemm_strided_batched(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zhemm_strided_batched(
                handle.as_raw(),
                side.into(),
                uplo.into(),
                m,
                n,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait HerkBatchedType {
    type ScalarType;

    unsafe fn rocblas_herk_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;

    unsafe fn rocblas_herk_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherk_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HerkBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;

    unsafe fn rocblas_herk_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const *const Self,
        lda: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherk_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub trait HerkStridedBatchedType {
    type ScalarType;

    unsafe fn rocblas_herk_strided_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkStridedBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;

    unsafe fn rocblas_herk_strided_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherk_strided_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HerkStridedBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;

    unsafe fn rocblas_herk_strided_batched(
        handle: &Handle,
        uplo: Fill,
        transA: Operation,
        n: i32,
        k: i32,
        alpha: &Self::ScalarType,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherk_strided_batched(
                handle.as_raw(),
                uplo.into(),
                transA.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

pub unsafe fn hemm_batched<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    B: *const *const T,
    ldb: i32,
    beta: &T,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HemmBatchedType,
{
    unsafe {
        T::rocblas_hemm_batched(
            handle,
            side,
            uplo,
            m,
            n,
            alpha,
            A,
            lda,
            B,
            ldb,
            beta,
            C,
            ldc,
            batch_count,
        )
    }
}

/// Strided batched Hermitian matrix-matrix multiplication
pub unsafe fn hemm_strided_batched<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    B: *const T,
    ldb: i32,
    stride_B: i64,
    beta: &T,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HemmStridedBatchedType,
{
    unsafe {
        T::rocblas_hemm_strided_batched(
            handle,
            side,
            uplo,
            m,
            n,
            alpha,
            A,
            lda,
            stride_A,
            B,
            ldb,
            stride_B,
            beta,
            C,
            ldc,
            stride_C,
            batch_count,
        )
    }
}

/// Batched Hermitian rank-k update
pub unsafe fn herk_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    transA: Operation,
    n: i32,
    k: i32,
    alpha: &R,
    A: *const *const T,
    lda: i32,
    beta: &R,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HerkBatchedType<ScalarType = R>,
{
    unsafe {
        T::rocblas_herk_batched(
            handle,
            uplo,
            transA,
            n,
            k,
            alpha,
            A,
            lda,
            beta,
            C,
            ldc,
            batch_count,
        )
    }
}

/// Strided batched Hermitian rank-k update
pub unsafe fn herk_strided_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    transA: Operation,
    n: i32,
    k: i32,
    alpha: &R,
    A: *const T,
    lda: i32,
    stride_A: i64,
    beta: &R,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HerkStridedBatchedType<ScalarType = R>,
{
    unsafe {
        T::rocblas_herk_strided_batched(
            handle,
            uplo,
            transA,
            n,
            k,
            alpha,
            A,
            lda,
            stride_A,
            beta,
            C,
            ldc,
            stride_C,
            batch_count,
        )
    }
}

/// Hermitian rank-k update with two matrices
///
/// Computes the matrix-matrix operation:
///
/// C := alpha * op(A) * op(B)^H + beta * C
///
/// This routine should only be used when the result of op(A)*op(B)^H will be Hermitian.
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part of C is used
/// * `trans` - Operation op(A) that is non-or-conjugate transpose
/// * `n` - Number of rows and columns of matrix C
/// * `k` - Number of columns of op(A) and op(B)
/// * `alpha` - Scalar alpha
/// * `A` - Buffer storing matrix A
/// * `lda` - Leading dimension of matrix A
/// * `B` - Buffer storing matrix B
/// * `ldb` - Leading dimension of matrix B
/// * `beta` - Scalar beta
/// * `C` - Buffer storing matrix C
/// * `ldc` - Leading dimension of matrix C
pub unsafe fn herkx<T, R>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    beta: &R,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: HerkxType<ScalarType = R>,
{
    unsafe {
        T::rocblas_herkx(
            handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
        )
    }
}

/// Batched Hermitian rank-k update with two matrices
pub unsafe fn herkx_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const *const T,
    lda: i32,
    B: *const *const T,
    ldb: i32,
    beta: &R,
    C: *const *mut T,
    ldc: i32,
    batch_count: i32,
) -> Result<()>
where
    T: HerkxBatchedType<ScalarType = R>,
{
    unsafe {
        T::rocblas_herkx_batched(
            handle,
            uplo,
            trans,
            n,
            k,
            alpha,
            A,
            lda,
            B,
            ldb,
            beta,
            C,
            ldc,
            batch_count,
        )
    }
}

/// Strided batched Hermitian rank-k update with two matrices
pub unsafe fn herkx_strided_batched<T, R>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    stride_A: i64,
    B: *const T,
    ldb: i32,
    stride_B: i64,
    beta: &R,
    C: *mut T,
    ldc: i32,
    stride_C: i64,
    batch_count: i32,
) -> Result<()>
where
    T: HerkxStridedBatchedType<ScalarType = R>,
{
    unsafe {
        T::rocblas_herkx_strided_batched(
            handle,
            uplo,
            trans,
            n,
            k,
            alpha,
            A,
            lda,
            stride_A,
            B,
            ldb,
            stride_B,
            beta,
            C,
            ldc,
            stride_C,
            batch_count,
        )
    }
}

/// Trait for types that can be used with herkx
pub trait HerkxType {
    type ScalarType;

    unsafe fn rocblas_herkx(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

impl HerkxType for ffi::rocblas_float_complex {
    type ScalarType = f32;

    unsafe fn rocblas_herkx(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherkx(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HerkxType for ffi::rocblas_double_complex {
    type ScalarType = f64;

    unsafe fn rocblas_herkx(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherkx(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with herkx_batched
pub trait HerkxBatchedType {
    type ScalarType;

    unsafe fn rocblas_herkx_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkxBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;

    unsafe fn rocblas_herkx_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherkx_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HerkxBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;

    unsafe fn rocblas_herkx_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const *const Self,
        lda: i32,
        B: *const *const Self,
        ldb: i32,
        beta: &Self::ScalarType,
        C: *const *mut Self,
        ldc: i32,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherkx_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Trait for types that can be used with herkx_strided_batched
pub trait HerkxStridedBatchedType {
    type ScalarType;

    unsafe fn rocblas_herkx_strided_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()>;
}

impl HerkxStridedBatchedType for ffi::rocblas_float_complex {
    type ScalarType = f32;

    unsafe fn rocblas_herkx_strided_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_cherkx_strided_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

impl HerkxStridedBatchedType for ffi::rocblas_double_complex {
    type ScalarType = f64;

    unsafe fn rocblas_herkx_strided_batched(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        stride_A: i64,
        B: *const Self,
        ldb: i32,
        stride_B: i64,
        beta: &Self::ScalarType,
        C: *mut Self,
        ldc: i32,
        stride_C: i64,
        batch_count: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::rocblas_zherkx_strided_batched(
                handle.as_raw(),
                uplo.into(),
                trans.into(),
                n,
                k,
                alpha,
                A,
                lda,
                stride_A,
                B,
                ldb,
                stride_B,
                beta,
                C,
                ldc,
                stride_C,
                batch_count,
            )
        };
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Note: `Diagonal` is defined in `super::types` and imported above.

//==============================================================================
// SYMM - Symmetric matrix-matrix multiplication
//==============================================================================

/// Symmetric matrix-matrix multiplication
///
/// C := alpha * A * B + beta * C  if side == Side::Left
/// C := alpha * B * A + beta * C  if side == Side::Right
///
/// where alpha and beta are scalars, A is a symmetric matrix, and B and C are m by n matrices.
pub unsafe fn symm<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    beta: &T,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: SymmType,
{
    unsafe {
        T::rocblas_symm(
            handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc,
        )
    }
}

pub trait SymmType {
    unsafe fn rocblas_symm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

macro_rules! impl_symm {
    ($t:ty, $func:path) => {
        impl SymmType for $t {
            unsafe fn rocblas_symm(
                handle: &Handle,
                side: Side,
                uplo: Fill,
                m: i32,
                n: i32,
                alpha: &Self,
                A: *const Self,
                lda: i32,
                B: *const Self,
                ldb: i32,
                beta: &Self,
                C: *mut Self,
                ldc: i32,
            ) -> Result<()> {
                let status = unsafe {
                    $func(
                        handle.as_raw(),
                        side.into(),
                        uplo.into(),
                        m,
                        n,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc,
                    )
                };
                if status != ffi::rocblas_status__rocblas_status_success {
                    return Err(Error::new(status));
                }
                Ok(())
            }
        }
    };
}

impl_symm!(f32, ffi::rocblas_ssymm);
impl_symm!(f64, ffi::rocblas_dsymm);
impl_symm!(ffi::rocblas_float_complex, ffi::rocblas_csymm);
impl_symm!(ffi::rocblas_double_complex, ffi::rocblas_zsymm);

//==============================================================================
// SYRK - Symmetric rank-k update
//==============================================================================

/// Symmetric rank-k update
///
/// C := alpha * A * A^T + beta * C  if transA == Operation::None
/// C := alpha * A^T * A + beta * C  if transA == Operation::Transpose
///
/// where alpha and beta are scalars, C is an n by n symmetric matrix, and A is an n by k matrix in
/// the first case and a k by n matrix in the second.
pub unsafe fn syrk<T>(
    handle: &Handle,
    uplo: Fill,
    trans_a: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    beta: &T,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: SyrkType,
{
    unsafe { T::rocblas_syrk(handle, uplo, trans_a, n, k, alpha, A, lda, beta, C, ldc) }
}

pub trait SyrkType {
    unsafe fn rocblas_syrk(
        handle: &Handle,
        uplo: Fill,
        trans_a: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

macro_rules! impl_syrk {
    ($t:ty, $func:path) => {
        impl SyrkType for $t {
            unsafe fn rocblas_syrk(
                handle: &Handle,
                uplo: Fill,
                trans_a: Operation,
                n: i32,
                k: i32,
                alpha: &Self,
                A: *const Self,
                lda: i32,
                beta: &Self,
                C: *mut Self,
                ldc: i32,
            ) -> Result<()> {
                let status = unsafe {
                    $func(
                        handle.as_raw(),
                        uplo.into(),
                        trans_a.into(),
                        n,
                        k,
                        alpha,
                        A,
                        lda,
                        beta,
                        C,
                        ldc,
                    )
                };
                if status != ffi::rocblas_status__rocblas_status_success {
                    return Err(Error::new(status));
                }
                Ok(())
            }
        }
    };
}

impl_syrk!(f32, ffi::rocblas_ssyrk);
impl_syrk!(f64, ffi::rocblas_dsyrk);
impl_syrk!(ffi::rocblas_float_complex, ffi::rocblas_csyrk);
impl_syrk!(ffi::rocblas_double_complex, ffi::rocblas_zsyrk);

//==============================================================================
// SYR2K - Symmetric rank-2k update
//==============================================================================

/// Symmetric rank-2k update
///
/// C := alpha * A * B^T + alpha * B * A^T + beta * C  if trans == Operation::None
/// C := alpha * A^T * B + alpha * B^T * A + beta * C  if trans == Operation::Transpose
///
/// where alpha and beta are scalars, C is an n by n symmetric matrix, and A and B are n by k
/// matrices in the first case and k by n matrices in the second.
pub unsafe fn syr2k<T>(
    handle: &Handle,
    uplo: Fill,
    trans: Operation,
    n: i32,
    k: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    beta: &T,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: Syr2kType,
{
    unsafe {
        T::rocblas_syr2k(
            handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
        )
    }
}

pub trait Syr2kType {
    unsafe fn rocblas_syr2k(
        handle: &Handle,
        uplo: Fill,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        beta: &Self,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

macro_rules! impl_syr2k {
    ($t:ty, $func:path) => {
        impl Syr2kType for $t {
            unsafe fn rocblas_syr2k(
                handle: &Handle,
                uplo: Fill,
                trans: Operation,
                n: i32,
                k: i32,
                alpha: &Self,
                A: *const Self,
                lda: i32,
                B: *const Self,
                ldb: i32,
                beta: &Self,
                C: *mut Self,
                ldc: i32,
            ) -> Result<()> {
                let status = unsafe {
                    $func(
                        handle.as_raw(),
                        uplo.into(),
                        trans.into(),
                        n,
                        k,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc,
                    )
                };
                if status != ffi::rocblas_status__rocblas_status_success {
                    return Err(Error::new(status));
                }
                Ok(())
            }
        }
    };
}

impl_syr2k!(f32, ffi::rocblas_ssyr2k);
impl_syr2k!(f64, ffi::rocblas_dsyr2k);
impl_syr2k!(ffi::rocblas_float_complex, ffi::rocblas_csyr2k);
impl_syr2k!(ffi::rocblas_double_complex, ffi::rocblas_zsyr2k);

//==============================================================================
// TRMM - Triangular matrix-matrix multiplication
//==============================================================================

/// Triangular matrix-matrix multiplication
///
/// C := alpha * op(A) * B  if side == Side::Left
/// C := alpha * B * op(A)  if side == Side::Right
///
/// where alpha is a scalar, B and C are m by n matrices, and A is a triangular matrix that is
/// optionally transposed and/or unit triangular.
pub unsafe fn trmm<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    trans_a: Operation,
    diag: Diagonal,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *const T,
    ldb: i32,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: TrmmType,
{
    unsafe {
        T::rocblas_trmm(
            handle, side, uplo, trans_a, diag, m, n, alpha, A, lda, B, ldb, C, ldc,
        )
    }
}

pub trait TrmmType {
    unsafe fn rocblas_trmm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        trans_a: Operation,
        diag: Diagonal,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *const Self,
        ldb: i32,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

macro_rules! impl_trmm {
    ($t:ty, $func:path) => {
        impl TrmmType for $t {
            unsafe fn rocblas_trmm(
                handle: &Handle,
                side: Side,
                uplo: Fill,
                trans_a: Operation,
                diag: Diagonal,
                m: i32,
                n: i32,
                alpha: &Self,
                A: *const Self,
                lda: i32,
                B: *const Self,
                ldb: i32,
                C: *mut Self,
                ldc: i32,
            ) -> Result<()> {
                let status = unsafe {
                    $func(
                        handle.as_raw(),
                        side.into(),
                        uplo.into(),
                        trans_a.into(),
                        diag.into(),
                        m,
                        n,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        C,
                        ldc,
                    )
                };
                if status != ffi::rocblas_status__rocblas_status_success {
                    return Err(Error::new(status));
                }
                Ok(())
            }
        }
    };
}

impl_trmm!(f32, ffi::rocblas_strmm);
impl_trmm!(f64, ffi::rocblas_dtrmm);
impl_trmm!(ffi::rocblas_float_complex, ffi::rocblas_ctrmm);
impl_trmm!(ffi::rocblas_double_complex, ffi::rocblas_ztrmm);

//==============================================================================
// TRSM - Triangular solve with multiple right-hand sides
//==============================================================================

/// Triangular solve with multiple right-hand sides
///
/// Solves op(A) * X = alpha * B  if side == Side::Left
/// Solves X * op(A) = alpha * B  if side == Side::Right
///
/// where alpha is a scalar, X and B are m by n matrices, and A is a triangular matrix that is
/// optionally transposed and/or unit triangular. B is overwritten with the solution matrix X.
pub unsafe fn trsm<T>(
    handle: &Handle,
    side: Side,
    uplo: Fill,
    trans_a: Operation,
    diag: Diagonal,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    B: *mut T,
    ldb: i32,
) -> Result<()>
where
    T: TrsmType,
{
    unsafe {
        T::rocblas_trsm(
            handle, side, uplo, trans_a, diag, m, n, alpha, A, lda, B, ldb,
        )
    }
}

pub trait TrsmType {
    unsafe fn rocblas_trsm(
        handle: &Handle,
        side: Side,
        uplo: Fill,
        trans_a: Operation,
        diag: Diagonal,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
    ) -> Result<()>;
}

macro_rules! impl_trsm {
    ($t:ty, $func:path) => {
        impl TrsmType for $t {
            unsafe fn rocblas_trsm(
                handle: &Handle,
                side: Side,
                uplo: Fill,
                trans_a: Operation,
                diag: Diagonal,
                m: i32,
                n: i32,
                alpha: &Self,
                A: *const Self,
                lda: i32,
                B: *mut Self,
                ldb: i32,
            ) -> Result<()> {
                let status = unsafe {
                    $func(
                        handle.as_raw(),
                        side.into(),
                        uplo.into(),
                        trans_a.into(),
                        diag.into(),
                        m,
                        n,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                    )
                };
                if status != ffi::rocblas_status__rocblas_status_success {
                    return Err(Error::new(status));
                }
                Ok(())
            }
        }
    };
}

impl_trsm!(f32, ffi::rocblas_strsm);
impl_trsm!(f64, ffi::rocblas_dtrsm);
impl_trsm!(ffi::rocblas_float_complex, ffi::rocblas_ctrsm);
impl_trsm!(ffi::rocblas_double_complex, ffi::rocblas_ztrsm);

//==============================================================================
// GEAM - General matrix-matrix addition
//==============================================================================

/// General matrix-matrix addition
///
/// C := alpha * op(A) + beta * op(B)
///
/// where alpha and beta are scalars, and A, B, C are matrices, with op(A) an m by n matrix and
/// op(B) an m by n matrix.
pub unsafe fn geam<T>(
    handle: &Handle,
    trans_a: Operation,
    trans_b: Operation,
    m: i32,
    n: i32,
    alpha: &T,
    A: *const T,
    lda: i32,
    beta: &T,
    B: *const T,
    ldb: i32,
    C: *mut T,
    ldc: i32,
) -> Result<()>
where
    T: GeamType,
{
    unsafe {
        T::rocblas_geam(
            handle, trans_a, trans_b, m, n, alpha, A, lda, beta, B, ldb, C, ldc,
        )
    }
}

pub trait GeamType {
    unsafe fn rocblas_geam(
        handle: &Handle,
        trans_a: Operation,
        trans_b: Operation,
        m: i32,
        n: i32,
        alpha: &Self,
        A: *const Self,
        lda: i32,
        beta: &Self,
        B: *const Self,
        ldb: i32,
        C: *mut Self,
        ldc: i32,
    ) -> Result<()>;
}

macro_rules! impl_geam {
    ($t:ty, $func:path) => {
        impl GeamType for $t {
            unsafe fn rocblas_geam(
                handle: &Handle,
                trans_a: Operation,
                trans_b: Operation,
                m: i32,
                n: i32,
                alpha: &Self,
                A: *const Self,
                lda: i32,
                beta: &Self,
                B: *const Self,
                ldb: i32,
                C: *mut Self,
                ldc: i32,
            ) -> Result<()> {
                let status = unsafe {
                    $func(
                        handle.as_raw(),
                        trans_a.into(),
                        trans_b.into(),
                        m,
                        n,
                        alpha,
                        A,
                        lda,
                        beta,
                        B,
                        ldb,
                        C,
                        ldc,
                    )
                };
                if status != ffi::rocblas_status__rocblas_status_success {
                    return Err(Error::new(status));
                }
                Ok(())
            }
        }
    };
}

impl_geam!(f32, ffi::rocblas_sgeam);
impl_geam!(f64, ffi::rocblas_dgeam);
impl_geam!(ffi::rocblas_float_complex, ffi::rocblas_cgeam);
impl_geam!(ffi::rocblas_double_complex, ffi::rocblas_zgeam);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::DeviceMemory;

    fn dev(data: &[f32]) -> DeviceMemory<f32> {
        let mut m = DeviceMemory::<f32>::new(data.len()).unwrap();
        m.copy_from_host(data).unwrap();
        m
    }

    fn host(m: &DeviceMemory<f32>, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        m.copy_to_host(&mut v).unwrap();
        v
    }

    fn approx(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-4, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn test_symm() {
        let handle = Handle::new().unwrap();
        // Symmetric A = [[2, 1], [1, 3]] column-major, B = identity.
        let a = dev(&[2.0, 0.0, 1.0, 3.0]);
        let b = dev(&[1.0, 0.0, 0.0, 1.0]);
        let mut c = dev(&[0.0, 0.0, 0.0, 0.0]);
        unsafe {
            symm(
                &handle,
                Side::Left,
                Fill::Upper,
                2,
                2,
                &1.0,
                a.as_ptr().cast::<f32>(),
                2,
                b.as_ptr().cast::<f32>(),
                2,
                &0.0,
                c.as_ptr().cast::<f32>(),
                2,
            )
            .unwrap();
        }
        // C = A * I = A = [[2, 1], [1, 3]] column-major.
        approx(&host(&c, 4), &[2.0, 1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_syr2k() {
        let handle = Handle::new().unwrap();
        // A = [[1], [2]], B = [[3], [4]] column-major (n=2, k=1).
        let a = dev(&[1.0, 2.0]);
        let b = dev(&[3.0, 4.0]);
        let mut c = dev(&[0.0, 0.0, 0.0, 0.0]);
        unsafe {
            syr2k(
                &handle,
                Fill::Lower,
                Operation::None,
                2,
                1,
                &1.0,
                a.as_ptr().cast::<f32>(),
                2,
                b.as_ptr().cast::<f32>(),
                2,
                &0.0,
                c.as_ptr().cast::<f32>(),
                2,
            )
            .unwrap();
        }
        // A*B^T + B*A^T = [[6, 10], [10, 16]]; only the lower triangle is written.
        let out = host(&c, 4);
        assert!((out[0] - 6.0).abs() < 1e-4, "c[0,0] = {}", out[0]);
        assert!((out[1] - 10.0).abs() < 1e-4, "c[1,0] = {}", out[1]);
        assert!((out[3] - 16.0).abs() < 1e-4, "c[1,1] = {}", out[3]);
    }

    #[test]
    fn test_trmm() {
        let handle = Handle::new().unwrap();
        // Upper triangular A = [[2, 1], [0, 3]], B = identity, separate output C.
        let a = dev(&[2.0, 0.0, 1.0, 3.0]);
        let b = dev(&[1.0, 0.0, 0.0, 1.0]);
        let mut c = dev(&[0.0, 0.0, 0.0, 0.0]);
        unsafe {
            trmm(
                &handle,
                Side::Left,
                Fill::Upper,
                Operation::None,
                Diagonal::NonUnit,
                2,
                2,
                &1.0,
                a.as_ptr().cast::<f32>(),
                2,
                b.as_ptr().cast::<f32>(),
                2,
                c.as_ptr().cast::<f32>(),
                2,
            )
            .unwrap();
        }
        // C = A * I = A = [[2, 1], [0, 3]] column-major.
        approx(&host(&c, 4), &[2.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_trsm() {
        let handle = Handle::new().unwrap();
        // Solve A X = B with A = [[2, 1], [0, 3]] and B = A; X = identity. B is in-place.
        let a = dev(&[2.0, 0.0, 1.0, 3.0]);
        let mut b = dev(&[2.0, 0.0, 1.0, 3.0]);
        unsafe {
            trsm(
                &handle,
                Side::Left,
                Fill::Upper,
                Operation::None,
                Diagonal::NonUnit,
                2,
                2,
                &1.0,
                a.as_ptr().cast::<f32>(),
                2,
                b.as_ptr().cast::<f32>(),
                2,
            )
            .unwrap();
        }
        approx(&host(&b, 4), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_geam() {
        let handle = Handle::new().unwrap();
        // C = A + B, A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] column-major.
        let a = dev(&[1.0, 3.0, 2.0, 4.0]);
        let b = dev(&[5.0, 7.0, 6.0, 8.0]);
        let mut c = dev(&[0.0, 0.0, 0.0, 0.0]);
        unsafe {
            geam(
                &handle,
                Operation::None,
                Operation::None,
                2,
                2,
                &1.0,
                a.as_ptr().cast::<f32>(),
                2,
                &1.0,
                b.as_ptr().cast::<f32>(),
                2,
                c.as_ptr().cast::<f32>(),
                2,
            )
            .unwrap();
        }
        // C = [[6, 8], [10, 12]] column-major.
        approx(&host(&c, 4), &[6.0, 10.0, 8.0, 12.0]);
    }
}
