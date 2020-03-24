// read_image(output float image[w][h][channels], output float y[m], param str path="data.txt", param str type="csv")
// {
//     index i[0:w-1], j[0:h-1], k[0:channels-1];
//
//     lines = fread(path, type);
//     image[i][j][k] = float(lines[i + j*w + channels*w*h]);
//     val = float(lines[m]);
//     y[j] = j == val ? 1.0 : 0.0;
// }


avg_pool2d(input float in[n][ic][h][w], output float out[n][ic][oh][ow], param int strides=2, param int pool_size=2, param int padding=0){

    index b[0:n-1], c[0:ic-1], y[0:oh-1], x[0:ow-1];
    index m[0:pool_size-1], i[0:pool_size-1];
    index y_pad[padding:oh + padding -1], x_pad[padding:ow + padding -1];

    padded_input[b][c][y][x] = 0;
    padded_input[b][c][y_pad][x_pad] = in[b][c][y_pad - padding][x_pad - padding];
    out[b][c][y][x] = (1/(pool_size^2))*sum[m][i](padded_input[b][c][strides*y + m][strides*x + n]);

}

batch_flatten(input float in[k][m][n][p], output float out[k][l]) {
    index i[0:l-1], j[0:k-1];
    out[j][i] = in[ i%m ][floor(i/m)%n][floor(i/(m%n))];
}
dense(input float in[n][m],
                        state float w[m][p],
                        output float out[n][p]){
    float a[n][p];

    index i[0:n-1], j[0:p-1], k[0:m-1];
//    out[i][j] = sum[k](in[i][k]*w[k][j]) + b[j];
    out[i][j] = sum[k](in[i][k]*w[k][j]);
}
sigmoid(input float in[n][m], output float out[n][m]){
  index i[0:n-1], j[0:m-1];

  out[i][j] = 1.0 / (1.0 + e()^(-in[i][j]));
}

relu(input float in[n][m], output float out[n][m]) {
    index i[0:n-1], j[0:m-1];
    out[i][j] = in[i][j] > 0 ? 1.0: 0.0;

}
softmax(input float y[n][m], output float out[m]){
    index i[0:m-1], j[0:m-1];

    out[i] = e()^(y[0][i])/sum[j](e()^(y[0][j]));

}
store_model(input float model[m], param str type="csv", param str model_path="model.txt"){

     fwrite(model, model_path, type);
 }

conv2d(input float x[n][ic][h][w], state float kernels[oc][ic][kh][kw],
                        output float result[n][oc][oh][ow], param int padding=0, param int strides=1){

        //Compute padding needed for the image
    index b[0:n-1], c[0:oc-1], y[0:oh-1], i[0:ow-1];
    index dy[0:kh-1], dx[0:kw-1], k[0:ic-1];
    index y_pad[padding:oh + padding -1], x_pad[padding:ow + padding -1];

    padded_input[b][k][y][i] = 0;
    padded_input[b][k][y_pad][x_pad] = x[b][c][y_pad - padding][x_pad - padding];

    result[b][c][y][i] = sum[dy][dx][ic](padded_input[b][k][strides*i + dx][strides*y + dy]*kernels[c][k][dy][dx]);
}
main(){
    float image[1][1][32][32], y[10], y_pred[10], c1_out[1][6][28][28], s2_out[1][6][14][14], c3_out[1][16][10][10], s4_out[1][16][5][5],s4_batch_flattened[1][400],c5_out[1][120], c6_out[1][84], c7_out[1][10];
    float f1[6][1][5][5], f2[16][6][5][5], f3[120][400], f4[84][120], f5[10][84];
    float b4[120], b5[10];
    test = 0;
    // read_image(image, y);
    conv2d(image, f1, c1_out, test);
    avg_pool2d(c1_out, s2_out);
    conv2d(s2_out, f2, c3_out);
    avg_pool2d(c3_out, s4_out);
    batch_flatten(s4_out,s4_batch_flattened);
    dense(s4_batch_flattened,f3,c5_out);
    dense(c5_out,f4, c6_out);
    dense(c6_out,f5, c7_out);
    softmax(c7_out, y_pred);
//    store_model(b5);

}