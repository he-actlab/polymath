import codegen as c


class Includes(object):

    include_list = ["math.h", "limits.h", "stdio.h", "float.h"]
    def __init__(self, header_name, external_includes=None):
        self.include_code = []
        self.include_code.append(c.Include(header_name, system=False))
        if external_includes:
            for e in external_includes:
                incl = c.Include(e, system=False)
                self.include_code.append(incl)

        self.add_system()

    def add_system(self):
        for i in self.include_list:
            incl = c.Include(i, system=True)
            self.include_code.append(incl)


