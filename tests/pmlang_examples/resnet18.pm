

avg_pool2d(input float in[n][ic][h][w], output float out[n][ic][oh][ow], param int strides=2, param int pool_size=2, param int padding=0){

    index b[0:n-1], c[0:ic-1], y[0:oh-1], x[0:ow-1];
    index m[0:pool_size-1], i[0:pool_size-1];
    index y_pad[padding:oh + padding -1], x_pad[padding:ow + padding -1];

    padded_input[b][c][y][x] = 0;
    padded_input[b][c][y_pad][x_pad] = x[b][c][y_pad - padding][x_pad - padding];
    out[b][c][y][x] = (1/(pool_size^2))*sum[m][i](in[b][c][strides*y + m][strides*x + n]);

}

max_pool2d(input float in[n][ic][h][w], output float out[n][ic][oh][ow], param int strides=2, param int pool_size=2, param int padding=0){

    index b[0:n-1], c[0:ic-1], y[0:oh-1], x[0:ow-1];
    index m[0:pool_size-1], i[0:pool_size-1];
    index y_pad[padding:oh + padding -1], x_pad[padding:ow + padding -1];

    padded_input[b][c][y][x] = 0;
    padded_input[b][c][y_pad][x_pad] = x[b][c][y_pad - padding][x_pad - padding];
    out[b][c][y][x] = max[m][i](in[b][c][strides*y + m][strides*x + n]);
}

batch_flatten(input float in[k][m][n][p], output float out[k][l]) {
    index i[0:l-1], j[0:k-1];
    out[j][i] = in[ i%m ][floor(i/m)%n][floor(i/(m%n))];
}
dense(input float in[n][m],
                        state float weights[m][p],
                        output float out[n][p]){
    float a[n][p];

    index i[0:n-1], j[0:p-1], k[0:m-1];
    out[i][j] = sum[k](in[i][k]*weights[k][j]);
}
sigmoid(input float in[n][m], output float out[n][m]){
  index i[0:n-1], j[0:m-1];

  out[i][j] = 1.0 / (1.0 + e()^(-in[i][j]));
}

relu(input float in[n][m][l][o], output float out[n][m][l][o]) {
    index i[0:n-1], j[0:m-1], k[0:l-1], p[0:o-1];
    out[i][j][k][p] = in[i][j][k][p] > 0 ? 1.0: 0.0;

}
softmax(input float y[n][m], output float out[1][m]){
    index i[0:m-1], j[0:m-1];

    out[0][i] = e()^(y[0][i])/sum[j](e()^(y[0][j]));

}
store_model(input float model[m], param str type="csv", param str model_path="model.txt"){

     fwrite(model, model_path, type);
 }

conv2d(input float x[n][ic][h][w], state float kernels[oc][ic][kh][kw],
                        output float result[n][oc][oh][ow],param int padding=0, param int strides=1){

        //Compute padding needed for the image
    index b[0:n-1], c[0:oc-1], y[0:oh-1], i[0:ow-1];
    index dy[0:kh-1], dx[0:kw-1], k[0:ic-1];
    index y_pad[padding:oh + padding -1], x_pad[padding:ow + padding -1];

    padded_input[b][k][y][i] = 0;
    padded_input[b][k][y_pad][x_pad] = x[b][c][y_pad - padding][x_pad - padding];

    result[b][c][y][i] = sum[dy][dx][ic](padded_input[b][k][strides*i + dx][strides*y + dy]*kernels[c][k][dy][dx]);
}

batch_norm(input float data[n][ic][h][w],
                     state float moving_mean[ic], state float moving_var[ic],
                     output float out[n][ic][h][w], state float gamma[ic],
                     state float beta[ic], param bool scale=1, param float epsilon=0.00002){

    index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
    momentum = 0.2;
    data_mean[j] = sum[i][y][x](data[i][j][y][x]) / (n*h*w);
    data_var[j] = sum[i][y][x]((data[i][j][y][x] - data_mean[j])^2) / ((n*h*w) - 1);
    out[i][j][y][x] = ((data[i][j][y][x] - data_mean[j])/(data_var[j] + epsilon)^(0.5))*gamma[j] + beta[j];
    moving_mean[j] = moving_mean[j] * momentum  + data_mean[j]*(1-momentum);
    moving_var[j] = moving_var[j] * momentum  + data_var[j]*(1-momentum);
}


res_stage(input float data[n][ic][h][w], output float data_out[ni][oc][oh][ow], param int strides=1, param int padding=1, param int kernel_size=3){

    // first layer edges
    float bn1_out[n][ic][h][w],bn1_moving_mean[ic],bn1_moving_var[ic],bn1_gamma[ic], bn1_beta[ic];
    float a1[n][ic][h][w];
    float conv1_weight[oc][ic][kernel_size][kernel_size], conv1_out[n][oc][oh][ow];

    // Second layer edges
    float bn2_out[n][oc][oh][ow], bn2_moving_mean[oc], bn2_moving_var[oc],bn2_gamma[oc], bn2_beta[oc];
    float a2[n][oc][oh][ow];

    float conv2_weight[oc][ic][kernel_size][kernel_size], conv2_out[n][oc][oh][ow];

    // Shortcut
    float sc1_weight[oc][ic][1][1], sc1_out[n][oc][oh][ow];


    // Layer 1
    batch_norm(data, bn1_moving_mean, bn1_moving_var, bn1_out, bn1_gamma, bn1_beta);
    relu(bn1_out, a1);
    conv2d(a1, conv1_weight, conv1_out, padding, strides);

    // Layer 2
    batch_norm(conv1_out, bn2_moving_mean, bn2_moving_var, bn2_out, bn2_gamma, bn2_beta);
    relu(bn2_out, a2);
    conv2d(a2, conv2_weight, conv2_out, padding, 1);

    // Shortcut
    add_elementwise(data, conv2_out, data_out);



}

res_stage_init(input float data[n][ic][h][w], output float data_out[ni][oc][oh][ow], param int strides=1, param int padding=1, param int kernel_size=3){

    // first layer edges
    float bn1_out[n][ic][h][w],bn1_moving_mean[ic],bn1_moving_var[ic],bn1_gamma[ic], bn1_beta[ic];
    float a1[n][ic][h][w];
    float conv1_weight[oc][ic][kernel_size][kernel_size], conv1_out[n][oc][oh][ow];

    // Second layer edges
    float bn2_out[n][oc][oh][ow], bn2_moving_mean[oc], bn2_moving_var[oc],bn2_gamma[oc], bn2_beta[oc];
    float a2[n][oc][oh][ow];
    float conv2_weight[oc][oc][kernel_size][kernel_size], conv2_out[n][oc][oh][ow];

    // Shortcut
    float sc1_weight[oc][ic][1][1], sc1_out[n][oc][oh][ow];


    // Layer 1
    batch_norm(data, bn1_moving_mean, bn1_moving_var, bn1_out, bn1_gamma, bn1_beta);
    relu(bn1_out, a1);
    conv2d(a1, conv1_weight, conv1_out, padding, strides);

    // Layer 2
    batch_norm(conv1_out, bn2_moving_mean, bn2_moving_var, bn2_out, bn2_gamma, bn2_beta);
    relu(bn2_out, a2);
    conv2d(a2, conv2_weight, conv2_out, padding, 1);

    // Shortcut
    conv2d(a1, sc1_weight, sc1_out, 0, strides);

    add_elementwise(conv2_out, sc1_out, data_out);


}
add_elementwise(input float op1[n][ic][h][w], input float op2[n][ic][h][w], output float out[n][ic][h][w]) {

        index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
        out[i][j][y][x] = op1[i][j][y][x] + op2[i][j][y][x];
}

global_pool(input float data[n][ic][h][w], output float out[n][ic][h][w]){
        index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
        out[i][j][0][0] = 1/(h*w) * sum[y][x](data[i][j][y][x]);

}
add_bias(input float in[y][x], state float bias[x], output float out[y][x]){
    index i[0:y-1],j[0:x-1];

    out[i][j] = in[i][j] + bias[j];

}
main(){
    // Input image
      float data[1][3][224][224];
    // Start Layers

    // Batch Norm1
    float bn1_out[1][3][224][224],bn1_moving_mean[3],bn1_moving_var[3],bn1_gamma[3], bn1_beta[3];
    // Conv1
    float conv1_weight[64][3][7][7], conv1_out[1][64][112][112];

    //Batch Norm2
    float bn2_out[1][64][122][122],bn2_moving_mean[64],bn2_moving_var[64],bn2_gamma[64], bn2_beta[64];
    //Relu1
    float a1[1][64][122][122];
    // Pool1
    float mp1[1][64][56][56];

    //Stage 1
    float stage1_unit1_out[1][64][56][56], stage1_unit2_out[1][64][56][56];

    //Stage 2
    float stage2_unit1_out[1][128][28][28], stage2_unit2_out[1][128][28][28];

    //Stage 3
    float stage3_unit1_out[1][256][14][14], stage3_unit2_out[1][256][14][14];

    //Stage 4
    float stage4_unit1_out[1][512][7][7], stage4_unit2_out[1][512][7][7];

    //Final Stage
    float bn3_out[1][512][7][7],bn3_moving_mean[512],bn3_moving_var[512],bn3_gamma[512], bn3_beta[512];
    float a2[1][512][7][7];
    float gp1[1][512][7][7];
    float bf1[1][512];
    float fc1_out[1][1000], fc1_weight[1000][512], fc1_bias[1000], fc1_bias_out[1][1000];
    float y_out[1][1000];

    // Output softmax

    batch_norm(data, bn1_moving_mean, bn1_moving_var, bn1_out, bn1_gamma, bn1_beta, 0);
    conv2d(bn1_out,conv1_weight, conv1_out, 3,2);
    batch_norm(conv1_out, bn2_moving_mean, bn2_moving_var, bn2_out, bn2_gamma, bn2_beta);
    relu(bn2_out, a1);
    max_pool2d(a1,mp1,2,3, 1);

    //stage 1
    res_stage_init(mp1, stage1_unit1_out);
    res_stage(stage1_unit1_out, stage1_unit2_out);

    //stage 2
    res_stage_init(stage1_unit2_out, stage2_unit1_out,2);
    res_stage(stage2_unit1_out, stage2_unit2_out);

    //stage 3
    res_stage_init(stage2_unit2_out, stage3_unit1_out,2);
    res_stage(stage3_unit1_out, stage3_unit2_out);

    //stage 4
    res_stage_init(stage3_unit2_out, stage4_unit1_out,2);
    res_stage(stage4_unit1_out, stage4_unit2_out);

    //Final Stage
    batch_norm(stage4_unit2_out,bn3_moving_mean, bn3_moving_var, bn3_out, bn3_gamma, bn3_beta);
    relu(bn3_out, a2);

    // Fix this definition
    global_pool(a2, gp1);
    batch_flatten(gp1, bf1);
    dense(bf1, fc1_weight, fc1_out);
    add_bias(fc1_out, fc1_bias, fc1_bias_out);
    softmax(fc1_bias_out, y_out);



}