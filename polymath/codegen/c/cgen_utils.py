import codegen as c
import hdfg.hdfgutils as utils
import logging
def create_declaration(edge, edge_map):
    expr_type = utils.get_attribute_value(edge.attributes['type'])
    vid = edge.vid
    var = create_value(expr_type, vid)

    vid_dims = utils.get_attribute_value(edge_map[vid].attributes['dimensions'])
    expr_dims = utils.get_attribute_value(edge.attributes['dimensions'])

    expr_attributes = list(edge.attributes)

    if len(vid_dims) > 0:
        for d in vid_dims:
            var = c.ArrayOf(var, d)
    elif len(expr_dims) > 0:
        for d in expr_dims:
            var = c.ArrayOf(var, d)
    elif edge.iid != '':
        iid = edge.iid
        bounds, bounds_dict = get_dims(iid, edge_map)
        for b in bounds:
            lower = bounds_dict[b][0]
            upper = bounds_dict[b][1]
            size = '(' +'(' + upper + ')' + '-' + lower +')' + '+1'
            var = c.ArrayOf(var, size)
    return vid, var


def create_value(dtype, name):
    if dtype == 'str':
        val = c.Pointer(c.Value('char', name))
    else:
        val = c.Value(dtype, name)
    return val

def is_string(var):
    if var[0] == '"' and var[-1] == '"':
        return True
    else:
        return False

def is_number(var):
    try:
        float(var)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(var)
        return True
    except (TypeError, ValueError):
        pass
    return False


def is_literal(var):
    str  = is_string(var)
    num = is_number(var)
    if str or num:
        return True
    else:
        return False


def get_index_dims(index_id, edge_map):
    index = edge_map[index_id]
    bounds = []
    bounds_dict = {}
    dims = utils.get_attribute_value(index.attributes['dimensions'])
    if len(dims) > 0:
        for d in dims:
            vid_type = utils.get_attribute_value(edge_map[d].attributes['vtype'])

            if vid_type not in ['scalar', 'var']:
                bounds.append(d)
                dim_edge = edge_map[d]
                if 'lower' not in list(dim_edge.attributes) or  'upper' not in list(dim_edge.attributes):
                    logging.error(f"Bounds not available for dimension {d}, variable {index_id}")
                    exit(1)
                else:
                    lower = utils.get_attribute_value(dim_edge.attributes['lower'])
                    upper = utils.get_attribute_value(dim_edge.attributes['upper'])
                    size = '(' +'(' + upper + ')' + '-' + lower +')' + '+1'
                    bounds_dict[d] = [lower, upper, size]
    else:
        bounds.append(index.name)
        lower = utils.get_attribute_value(index.attributes['lower'])
        upper = utils.get_attribute_value(index.attributes['upper'])
        size = '(' + '(' + upper + ')' + '-' + lower + ')' + '+1'
        bounds_dict[index.name] = [lower, upper, size]
    return bounds, bounds_dict

def get_dims(edge, edge_map):
    edge_dims = utils.get_attribute_value(edge.attributes['dimensions'])
    vid = edge.vid
    vid_dims = utils.get_attribute_value(edge_map[vid].attributes['dimensions'])
    attributes = list(edge.attributes)

    if len(vid_dims) > 0:
        return vid_dims
    elif len(edge_dims) > 0:
        return edge_dims
    elif edge.iid != '':
        iid = edge.iid
        vid_type = utils.get_attribute_value(edge_map[iid].attributes['vtype'])
        if vid_type not in ['scalar', 'var']:
            bounds, bounds_dict = get_index_dims(iid, edge_map)
            return [bounds_dict[b][2] for b in bounds]
        else:
            return [iid]
    else:
        return []

FUNCTION_MACRO_MAP = {"pi": "M_PI",
             "log": "log({val})",
             "log2": "log2({val})",
             "float": "FLOAT_CAST({val})",
             "int": "INT_CAST({val})",
             "bin": "BINARY_CAST({val})",
             "ceiling": "ceil({val})",
             "floor": "floor({val})",
             "e": "M_E",
             "fread": "FREAD({val})",
             "fwrite": "FWRITE({val})",
            "sin": "sin({val})",
            "sinh": "sinh({val})",
            "cos": "cos({val})",
            "cosh": "cosh({val})",
            "tan": "tan({val})",
            "tanh": "cosh({val})"
                      }



