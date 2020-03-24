
import codegen as c
class Index(object):

    def __init__(self,loop_body, bounds_dict, bounds):
        self.loop_body = loop_body
        self.bounds_dict = bounds_dict

        self.loop = self.create_loop(bounds)

    def create_loop(self, bounds):
        if len(bounds[1:]) == 0:
            it = bounds[0]
            return c.For("int {it} = {low}".format(it=it, low=self.bounds_dict[it][0]),
                         "{it} <= {high}".format(it=it, high=self.bounds_dict[it][1]),
                         "++{it}".format(it=it),c.Block(self.loop_body))
        else:
            it = bounds[0]
            return c.For("int {it} = {low}".format(it=it, low=self.bounds_dict[it][0]),
                         "{it} <= {high}".format(it=it, high=self.bounds_dict[it][1]),
                         "++{it}".format(it=it),c.Block([self.create_loop(bounds[1:])]))