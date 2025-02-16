extern crate ocl;
use ocl::{ProQue};
use std::time::Instant;

const src_1: &str = r#"
__kernel void matrix_multiply(__global float* A, __global float* B, __global float* C, const unsigned int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    for (int i = 0; i < N; i++) {
        C[row * N + col] += A[row * N + i] * B[i * N + col];
    }
}
"#;

const src_2: &str = r#"
__kernel void matrix_multiply(__global float* A, __global float* B, __global float* C, const unsigned int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B[i * N + col];
    }
    
    C[row * N + col] = sum;
}
"#;

const src_3: &str = r#"
__kernel void matrix_multiply(
    __global float* A, 
    __global float* B_T,
    __global float* C, 
    const unsigned int N
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    
    for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B_T[col * N + i];
    }
    
    C[row * N + col] = sum;
}
"#;

const src_4: &str = r#"
__kernel void matrix_multiply(
    __global float* A, 
    __global float* B_T,
    __global float* C, 
    const unsigned int N
) {
    // Tile size (assuming 8x8 workgroup size)
    const unsigned int TILE_SIZE = 8;

    // Thread indices within the workgroup
    int row = get_local_id(0);
    int col = get_local_id(1);

    // Global indices for A and B_T in the entire matrix
    int global_row = get_group_id(0) * TILE_SIZE + row;
    int global_col = get_group_id(1) * TILE_SIZE + col;

    // Local memory to store tiles of A and B_T
    __local float local_A[TILE_SIZE][TILE_SIZE];
    __local float local_B_T[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    
    // Loop through the tiles of A and B_T
    for (int k = 0; k < N / TILE_SIZE; k++) {
        // Load a block of A from global memory to local memory
        local_A[row][col] = A[global_row * N + (k * TILE_SIZE + col)];
        
        // Load a block of B_T (transposed B) from global memory to local memory
        local_B_T[row][col] = B_T[global_col * N + (k * TILE_SIZE + row)];

        // Synchronize threads to make sure all threads have finished loading data
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the multiplication and accumulation for the tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += local_A[row][i] * local_B_T[col][i];
        }

        // Synchronize to ensure all threads are done with the current computation before moving to the next
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result to the global memory (C matrix)
    if (global_row < N && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}
"#;

fn task() -> ocl::Result<()> {

    let m_n = 2048;
    let matrix_size = m_n * m_n;

    let mut m_a = vec![0.0f32; matrix_size];
    let mut m_b = vec![0.0f32; matrix_size];

    for i in 0..m_n {
        for j in 0..m_n {
            m_a[i * m_n + j] = (i as f32) + (j as f32);
            m_b[i * m_n + j] = (i as f32) - (j as f32);
        }
    }

    let pro_que = ProQue::builder()
        .src(src_4)
        .dims([m_n, m_n])
        .build()?;

    let buffer_a = pro_que.create_buffer::<f32>()?;
    let buffer_b = pro_que.create_buffer::<f32>()?;
    let buffer_c = pro_que.create_buffer::<f32>()?;

    buffer_a.write(&m_a).enq()?;
    buffer_b.write(&m_b).enq()?;

    let kernel = pro_que.kernel_builder("matrix_multiply")
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_c)
        .arg(m_n as u32)
        .local_work_size([8, 8])
        .build()?;

    let start = Instant::now();

    unsafe { kernel.enq()?; }

    let mut m_c = vec![0.0f32; matrix_size];
    buffer_c.read(&mut m_c).enq()?;

    let duration = start.elapsed();
    println!("Matrix multiplication took: {:?}", duration);

    println!("C[1][1] = {}", m_c[1 * m_n + 1]);

    Ok(())
}

fn main() {
    task().unwrap();
}