
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
    out[b][c][y][x] = max[m][i](in[b][c][strides*y + m][strides*i + n]);
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

relu(input float in[n][ic][h][w], output float out[n][ic][h][w]) {

        index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
        out[i][j][y][x] = in[i][j][y][x] > 0 ? in[i][j][y][x] : in[i][j][y][x];
}

leaky_relu(input float in[n][ic][h][w], output float out[n][ic][h][w], param float alpha=0.1) {

        index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
        out[i][j][y][x] = in[i][j][y][x] > 0 ? in[i][j][y][x] : in[i][j][y][x]*alpha;
}
softmax(input float y[n][m], output float out[m]){
    index i[0:m-1], j[0:m-1];

    out[i] = e()^(y[0][i])/sum[j](e()^(y[0][j]));

}
 store_model(input float model[m], param str type="csv", param str model_path="model.txt"){

     fwrite(model, model_path, type);
 }



batch_norm(input float data[n][ic][h][w], state float moving_mean[ic], state float moving_var[ic],
                     output float out[n][ic][h][w], state float gamma[ic], state float beta[ic],param bool scale=1, param float epsilon=0.00002){

    index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
    momentum = 0.2;
    data_mean[j] = sum[i][y][x]((data[i][j][y][x])) / (n*h*w);
    data_var[j] = sum[i][y][x]((data[i][j][y][x] - data_mean[j])^2) / (n*h*w - 1);
    out[i][j][y][x] = ((data[i][j][y][x] - data_mean[j])/(data_var[j] + epsilon)^(0.5))*gamma[j] + beta[j];
    moving_mean[j] = moving_mean[j] * momentum  + data_mean[j]*(1-momentum);
    moving_var[j] = moving_var[j] * momentum  + data_var[j]*(1-momentum);
}


add_elementwise(input float op1[n][ic][h][w], input float op2[n][ic][h][w], output float out[n][ic][h][w]) {

        index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
        out[i][j][y][x] = op1[i][j][y][x] + op2[i][j][y][x];
}

global_pool(input float data[n][ic][h][w], output float out[n][ic][h][w]){
        index i[0:n-1], j[0:ic-1], y[0:h-1], x[0:w-1];
        out[i][j][0][0] = 1/(h*w) * sum[y][x](data[i][j][y][x]);

}
add_bias(input float in[n][ic][h][w], state float bias[ic], output float out[n][ic][w][h]){
    index x[0:w-1],y[0:h-1], i[0:n-1], j[0:ic-1];

    out[i][j][y][x] = in[i][j][y][x] + bias[j];

}


yolo_layer(input float data[n][ic][h][w], output float data_out[ni][oc][oh][ow], param int strides=2, param int padding=1, param int kernel_size=3){


    // Conv edges
    float conv1_weight[oc][ic][kernel_size][kernel_size], conv1_out[n][oc][oh][ow];
    // Bias
    float c1_bias[oc], c1_bias_out[n][oc][oh][ow];

    // Batch Norm
    float bn1_out[n][oc][oh][ow],bn1_moving_mean[oc],bn1_moving_var[oc],bn1_gamma[oc], bn1_beta[oc];
    //Activation
    float a1[n][oc][oh][ow];

    conv2d(data, conv1_weight, conv1_out, padding, 1);
    add_bias(conv1_out, c1_bias,c1_bias_out);
    batch_norm(c1_bias_out, bn1_moving_mean, bn1_moving_var, bn1_out, bn1_gamma, bn1_beta);
    relu(bn1_out, a1);
    max_pool2d(a1, data_out,strides, 2);

}

yolo_layer_no_pool(input float data[n][ic][h][w], output float data_out[ni][oc][oh][ow], param int strides=1, param int padding=1, param int kernel_size=3){

    // Conv edges
    float conv1_weight[oc][ic][kernel_size][kernel_size], conv1_out[n][oc][oh][ow];
    // Bias
    float c1_bias[oc], c1_bias_out[n][oc][oh][ow];

    // Batch Norm
    float bn1_out[n][oc][oh][ow],bn1_moving_mean[oc],bn1_moving_var[oc],bn1_gamma[oc], bn1_beta[oc];
    //Activation
    float a1[n][oc][oh][ow];


    conv2d(data, conv1_weight, conv1_out, padding, 1);
    add_bias(conv1_out, c1_bias,c1_bias_out);
    batch_norm(c1_bias_out, bn1_moving_mean, bn1_moving_var, bn1_out, bn1_gamma, bn1_beta);
    relu(bn1_out, data_out);

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
main(){

    // Yolo 1
    float data[1][3][416][416], yolo1_out[1][16][208][208], yolo2_out[1][32][104][104], yolo3_out[1][64][52][52];
    float yolo4_out[1][128][26][26], yolo5_out[1][256][13][13], yolo6_out[1][512][12][12];

    // Yolo 2
    float yolo7_out[1][1024][12][12], yolo8_out[1][1024][12][12];

    // Final layer
    float conv0_weight[125][1024][1][1], conv0_out[1][125][12][12], b0_bias[125], conv1_bias_out[1][125][12][12];
    yolo_layer(data, yolo1_out);
    yolo_layer(yolo1_out, yolo2_out);
    yolo_layer(yolo2_out, yolo3_out);
    yolo_layer(yolo3_out, yolo4_out);
    yolo_layer(yolo4_out, yolo5_out);
    yolo_layer(yolo5_out, yolo6_out, 1);
    yolo_layer_no_pool(yolo6_out, yolo7_out);
    yolo_layer_no_pool(yolo7_out, yolo8_out);

    conv2d(yolo8_out, conv0_weight, conv0_out, 1);
    add_bias(conv0_out, b0_bias, conv1_bias_out);
}