
rec_model(input float x1[k], input float x2[k], input float r1[m], input float y1[m], input float r2[n], input float y2[n],
                    state float w1[m][k], state float w2[n][k], param int mu=1) {
//    m = 3;
//    n = 3;
//    k = 2;
    index i[0:m-1], j[0:n-1], l[0:k-1];
    h1[i] = sum[l](w1[i][l] * x2[l]) * r1[i];
    h2[j] = sum[l](x1[l] * w2[j][l]) * r2[j];
    d1[i] = h1[i] - y1[i];
    d2[j] = h2[j] - y2[j];
    g1[i][l] = d1[i] * x2[l];
    g2[j][l] = d2[j] * x1[l];
    w1[i][l] = w1[i][l] - 1.0 * g1[i][l];
    w2[j][l] = w2[j][l] - g2[j][l];


}


main()
{
    float x1_input[2], x2_input[2], r1_output[3], y1_output[3], r2_output[3], y2_output[3];
    float w1_model[3][2], w2_model[3][2];

    rec_model(x1_input, x2_input, r1_output, y1_output, r2_output, y2_output, w1_model, w2_model);

}
