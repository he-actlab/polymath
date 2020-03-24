//spring read_data(output float x[m], output float y, param str type="csv",
//    param str data_path="dataset1.txt")
//{
//
//    index i[0:m-1];
//
//    lines = fread(data_path, type, m+1);
//    x[i] = float(lines[i]);
//    y = float(lines[m]);
//
//
//}
svd(input float x[m], input float y, state float w[m], param float mu=1.0)
{
    index i[0:m-1];

    h = sum[i](w[i] * x[i]);
    c = y * h;

    ny = 0 - y;
    p = ((c > 1) * ny);
    gi[i] = p* x[i];

    // SGD added
    g[i] = mu * gi[i];
    w[i] = w[i] - g[i];
}

//training_process(output float w[m])
//{
//     float x[m], y;
//
//    read_data(x,y);
//    svd(x,y,w);
//}
//
//reservoir store_data(input float w[m],param str type="csv", param str model_path="model_path.txt")
//{
//    fwrite(w, model_path, type);
//}



main()
{
     float w_model[3], x_input[3], y_input;
    svd(x_input, y_input, w_model);

}
