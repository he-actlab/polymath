// Need to define
 training_record(output float x[m], output float y, param str type="csv", param str path="dataset1.txt")
 {
     index i[0:m-1];
     // Read a file, one line at a time to populate a FIFO queue
     lines[i] = fread(path, type, m+1, 1);

     // The arguments below will format the data prior to adding to the output
     // queues
     x[i] = float(lines[i]);
     y = float(lines[m]);
 }



linear_regression(input float x[m],input float y, state float w[m], param float mu=1.0)
{

    index i[0:m-1], j[0:m-1];

    // Pop values off of the queue and perform operations on them,
    // updating a maintained state flow

    h = sum[i](w[i] * x[i]);
    d = h - y;
    g[i] = d * x[i];

    // SGD added
     w[i] = w[i] - mu*g[i];

}


 trained_model(input float w[m], param str path="results.txt", param str type="csv"){

     fwrite(w, path, type);

 }

// Starting point in the program
main(input float x_input[784], output float y_output)
{
    float w_model[784];

    training_record(x_input, y_output,"csv");
    linear_regression(x_input, y_output, w_model);

     trained_model(w_model);
}
