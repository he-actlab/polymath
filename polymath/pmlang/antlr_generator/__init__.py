from __future__ import absolute_import


LOOPY_IF = "{prefix} = if({predicate}, {true_stmt}, {false_stmt})"
LOOPY_ASSIGN = "{prefix} = {expr}"
INDEX_DOMAIN = "{low} <= {var} <= {upper}"
ASSUMPTION_DOMAIN = "{var} >= {low}"
FULL_DOMAIN = "{{[{dom_names}]: {domains}}}"
