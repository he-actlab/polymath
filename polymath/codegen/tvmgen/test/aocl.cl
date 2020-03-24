__kernel void fused_nn_conv2d_nn_relu_1_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu) {
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int yy = 0; yy < 28; ++yy) {
      for (int xx = 0; xx < 28; ++xx) {
        compute[((yy * 28) + xx)] = 0.000000e+00f;
        for (int ry = 0; ry < 5; ++ry) {
          for (int rx = 0; rx < 5; ++rx) {
            compute[((yy * 28) + xx)] = (compute[((yy * 28) + xx)] + (placeholder[((((yy * 32) + (ry * 32)) + xx) + rx)] * placeholder1[(((ax1 * 25) + (ry * 5)) + rx)]));
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 28; ++ax2) {
      for (int ax3 = 0; ax3 < 28; ++ax3) {
        T_relu[(((ax1 * 784) + (ax2 * 28)) + ax3)] = max(compute[((ax2 * 28) + ax3)], 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_avg_pool2d_1_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 6; ++ax1) {
    for (int ax2 = 0; ax2 < 14; ++ax2) {
      for (int ax3 = 0; ax3 < 14; ++ax3) {
        tensor[(((ax1 * 196) + (ax2 * 14)) + ax3)] = 0.000000e+00f;
        for (int rv = 0; rv < 2; ++rv) {
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[(((ax1 * 196) + (ax2 * 14)) + ax3)] = (tensor[(((ax1 * 196) + (ax2 * 14)) + ax3)] + (placeholder[(((((ax1 * 784) + (ax2 * 56)) + (rv * 28)) + (ax3 * 2)) + rv1)] / ((float)4)));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_batch_flatten_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax0_ax1_fused_inner = 0; ax0_ax1_fused_inner < 400; ++ax0_ax1_fused_inner) {
    tensor[ax0_ax1_fused_inner] = placeholder[ax0_ax1_fused_inner];
  }
}

__kernel void fused_nn_dense_nn_relu_1_kernel0(__global float* restrict T_dense, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu) {
  for (int ax1 = 0; ax1 < 84; ++ax1) {
    T_dense[0] = 0.000000e+00f;
    for (int k = 0; k < 120; ++k) {
      T_dense[0] = (T_dense[0] + (placeholder[k] * placeholder1[((ax1 * 120) + k)]));
    }
    T_relu[ax1] = max(T_dense[0], 0.000000e+00f);
  }
}

__kernel void fused_nn_conv2d_nn_relu_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu) {
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int yy = 0; yy < 10; ++yy) {
      for (int xx = 0; xx < 10; ++xx) {
        compute[((yy * 10) + xx)] = 0.000000e+00f;
        for (int rc = 0; rc < 6; ++rc) {
          for (int ry = 0; ry < 5; ++ry) {
            for (int rx = 0; rx < 5; ++rx) {
              compute[((yy * 10) + xx)] = (compute[((yy * 10) + xx)] + (placeholder[(((((rc * 196) + (yy * 14)) + (ry * 14)) + xx) + rx)] * placeholder1[((((ax1 * 150) + (rc * 25)) + (ry * 5)) + rx)]));
            }
          }
        }
      }
    }
    for (int ax2 = 0; ax2 < 10; ++ax2) {
      for (int ax3 = 0; ax3 < 10; ++ax3) {
        T_relu[(((ax1 * 100) + (ax2 * 10)) + ax3)] = max(compute[((ax2 * 10) + ax3)], 0.000000e+00f);
      }
    }
  }
}

__kernel void fused_nn_avg_pool2d_kernel0(__global float* restrict tensor, __global float* restrict placeholder) {
  for (int ax1 = 0; ax1 < 16; ++ax1) {
    for (int ax2 = 0; ax2 < 5; ++ax2) {
      for (int ax3 = 0; ax3 < 5; ++ax3) {
        tensor[(((ax1 * 25) + (ax2 * 5)) + ax3)] = 0.000000e+00f;
        for (int rv = 0; rv < 2; ++rv) {
          for (int rv1 = 0; rv1 < 2; ++rv1) {
            tensor[(((ax1 * 25) + (ax2 * 5)) + ax3)] = (tensor[(((ax1 * 25) + (ax2 * 5)) + ax3)] + (placeholder[(((((ax1 * 100) + (ax2 * 20)) + (rv * 10)) + (ax3 * 2)) + rv1)] / ((float)4)));
          }
        }
      }
    }
  }
}

__kernel void fused_nn_dense_nn_relu_kernel0(__global float* restrict T_dense, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu) {
  for (int ax1 = 0; ax1 < 10; ++ax1) {
    T_dense[0] = 0.000000e+00f;
    for (int k = 0; k < 84; ++k) {
      T_dense[0] = (T_dense[0] + (placeholder[k] * placeholder1[((ax1 * 84) + k)]));
    }
    T_relu[ax1] = max(T_dense[0], 0.000000e+00f);
  }
}

__kernel void fused_nn_dense_nn_relu_2_kernel0(__global float* restrict T_dense, __global float* restrict placeholder, __global float* restrict placeholder1, __global float* restrict T_relu) {
  for (int ax1 = 0; ax1 < 120; ++ax1) {
    T_dense[0] = 0.000000e+00f;
    for (int k = 0; k < 400; ++k) {
      T_dense[0] = (T_dense[0] + (placeholder[k] * placeholder1[((ax1 * 400) + k)]));
    }
    T_relu[ax1] = max(T_dense[0], 0.000000e+00f);
  }
}

__kernel void fused_nn_softmax_kernel0(__global float* restrict tensor, __global float* restrict placeholder, __global float* restrict tensor1, __global float* restrict tensor2) {
  for (int ax1 = 0; ax1 < 10; ++ax1) {
    tensor[0] = -3.402823e+38f;
    for (int k1 = 0; k1 < 10; ++k1) {
      tensor[0] = max(tensor[0], placeholder[k1]);
    }
    tensor1[0] = 0.000000e+00f;
    for (int k2 = 0; k2 < 10; ++k2) {
      tensor1[0] = (tensor1[0] + exp((placeholder[k2] - tensor[0])));
    }
    tensor2[ax1] = (exp((placeholder[ax1] - tensor[0])) / tensor1[0]);
  }
}

