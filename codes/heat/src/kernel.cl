__kernel void next_step(
    __constant float* T_current,
    __global float* T_next,
    const float dt,
    const float dl,
    const float a0,
    const float a1,
    const float a2
) {
    int p = get_global_id(0);
    int q = get_global_id(1);
    
    int size_x = get_global_size(0);
    int size_y = get_global_size(1);
    
    if (p == 0 || q == 0) {
        T_next[p * size_y + q] = 100.0f;
    } else if (p == size_x - 1 || q == size_y - 1) {
        T_next[p * size_y + q] = 0.0f;
    } else {
        float current_T = T_current[p * size_y + q];
        
        float T_p_plus1 = T_current[(p + 1) * size_y + q];
        float T_p_minus1 = T_current[(p - 1) * size_y + q];
        float T_q_plus1 = T_current[p * size_y + (q + 1)];
        float T_q_minus1 = T_current[p * size_y + (q - 1)];
        
        float divider = dl * dl;
        float laplacian_x = (T_p_plus1 - 2.0f * current_T + T_p_minus1);
        float laplacian_y = (T_q_plus1 - 2.0f * current_T + T_q_minus1);
        
        float alpha = a0 + a1 * current_T + a2 * current_T * current_T;
        float delta_T = alpha * (laplacian_x + laplacian_y) * dt;

        T_next[p * size_y + q] = current_T + delta_T;
    }
}
