func_table = {
        'pi' : 2,
        'sum': 2,
        'norm' : 2,
        'gaussian' : 1,
        'sigmoid' : 1,
        'sig_sym' : 1,
        'log' : 1
}

group_ops = {
        'pi' : '*',
        'sum': '+',
        'norm' : '+',
        'gaussian' : '',
        'sigmoid' : '',
        'sig_sym' : '',
        'log' : ''
}

data_priorities = {
            "model_input": 5,
            "model_output": 5,
            "model": 5,
            None: 3,
            "gradient": 4,
            "constant": 2
        }