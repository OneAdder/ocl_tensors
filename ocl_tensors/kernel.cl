__kernel void scalar_sum(
    __global const real_number *m,
    const real_number n,
    __global real_number *res
){
    int gid = get_global_id(0);
    res[gid] = m[gid] + n;
}


__kernel void scalar_mult(
    __global const real_number *m,
    const real_number n,
    __global real_number *res
){
    int gid = get_global_id(0);
    res[gid] = m[gid] * n;
}


__kernel void inner_product(
    __global const real_number *m1,
    __global const real_number *m2,
    __global real_number *res
){
    int gid = get_global_id(0);
    res[gid] = m1[gid] * m2[gid];
}


__kernel void tensor_sum(
    __global const real_number *m1,
    __global const real_number *m2,
    __global real_number *res
){
    int gid = get_global_id(0);
    res[gid] = m1[gid] + m2[gid];
}


__kernel void matmul(
    int a_width, int b_width, int c_width,
    __global real_number* a_elements,
    __global real_number* b_elements,
    __global real_number* c_elements
){
    int global_row = get_global_id(1);
    int global_col = get_global_id(0);
    real_number c_value = 0;
    for(int i=0; i<a_width; ++i){
        c_value += a_elements[global_row * a_width + i]
            * b_elements[i * b_width + global_col];
    }
    c_elements[global_row * c_width + global_col] = c_value;
} 
