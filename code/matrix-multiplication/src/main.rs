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
            sum += local_A[row][i] * local_B_T[i][col];
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

const src_5: &str = r#"
__kernel void matrix_multiply(
    __constant float* A, 
    __constant float* B_T,
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
            sum += local_A[row][i] * local_B_T[i][col];
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

const src_6: &str = r#"
__kernel void matrix_multiply(
    __constant float* A, 
    __constant float* B_T,
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
    __local float local_A[TILE_SIZE][TILE_SIZE + 1];
    __local float local_B_T[TILE_SIZE][TILE_SIZE + 1];

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
            sum += local_A[row][i] * local_B_T[i][col];
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

const src_7: &str = r#"
__kernel void matrix_multiply(
    __global float* A, 
    __global float* B_T,
    __global float* C, 
    const unsigned int N
) {
    const unsigned int TILE_SIZE = 8;
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int global_row = get_group_id(0) * TILE_SIZE + row;
    const int global_col = get_group_id(1) * TILE_SIZE + col;

    // Local memory with transposed storage for B_T
    __local float local_A[TILE_SIZE][TILE_SIZE];
    __local float local_B_T[TILE_SIZE][TILE_SIZE];  // Stored transposed

    float8 sum_vec = (float8)(0.0f);
    
    for (int k = 0; k < (N + TILE_SIZE - 1)/TILE_SIZE; k++) {
        // Load tiles with boundary checks
        const int load_col = k * TILE_SIZE + col;
        const int load_row = k * TILE_SIZE + row;
        
        local_A[row][col] = (load_col < N) ? A[global_row * N + load_col] : 0.0f;
        local_B_T[col][row] = (load_row < N) ? B_T[global_col * N + load_row] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);

        float8 a_vec = vload8(0, &local_A[row][0]);
        float8 b_vec = vload8(0, &local_B_T[row][0]);
        sum_vec += a_vec * b_vec;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Horizontal sum
    float sum = dot(sum_vec.lo, (float4)(1.0f)) + dot(sum_vec.hi, (float4)(1.0f));
    
    if(global_row < N && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}
"#;

const src_8: &str = r#"
__kernel void matrix_multiply(
    __global float* A,
    __global float* B_T,
    __global float* C,
    const unsigned int N
) {
    const unsigned int TILE_SIZE = 8;
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int global_row = get_group_id(0) * TILE_SIZE + row;
    const int global_col = get_group_id(1) * TILE_SIZE + col;

    __local float local_A[2][TILE_SIZE][TILE_SIZE];
    __local float local_B_T[2][TILE_SIZE][TILE_SIZE];
    int current = 0;
    float8 sum_vec = (float8)(0.0f);

    int load_col = 0 * TILE_SIZE + col;
    int load_row = 0 * TILE_SIZE + row;
    local_A[current][row][col] = (load_col < N) ? A[global_row * N + load_col] : 0.0f;
    local_B_T[current][col][row] = (load_row < N) ? B_T[global_col * N + load_row] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        int next = 1 - current;

        if (k + 1 < (N + TILE_SIZE - 1) / TILE_SIZE) {
            int next_load_col = (k + 1) * TILE_SIZE + col;
            int next_load_row = (k + 1) * TILE_SIZE + row;
            local_A[next][row][col] = (next_load_col < N) ? A[global_row * N + next_load_col] : 0.0f;
            local_B_T[next][col][row] = (next_load_row < N) ? B_T[global_col * N + next_load_row] : 0.0f;
        }

        float8 a_vec = vload8(0, &local_A[current][row][0]);
        float8 b_vec = vload8(0, &local_B_T[current][row][0]);
        sum_vec += a_vec * b_vec;

        barrier(CLK_LOCAL_MEM_FENCE);
        current = next;
    }

    if (global_row < N && global_col < N) {
        float sum = dot(sum_vec.lo, (float4)(1.0f)) + dot(sum_vec.hi, (float4)(1.0f));
        C[global_row * N + global_col] = sum;
    }
}
"#;

fn task() -> ocl::Result<()> {

    let m_n = 2048;
    let matrix_size: usize = m_n * m_n;

    let mut m_a = vec![0.0f32; matrix_size];
    let mut m_b = vec![0.0f32; matrix_size];

    for i in 0..m_n {
        for j in 0..m_n {
            m_a[i * m_n + j] = (i as f32) + (j as f32);
            m_b[i * m_n + j] = (i as f32) - (j as f32);
        }
    }

    let pro_que = ProQue::builder()
        .src(src_8)
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