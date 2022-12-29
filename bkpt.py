#Tom Wallace
#6482558
#Brock University
#An object class used to handle breakpoints during breakpoint detection/classification
class bkpt:
    def __init__(self, array, last):
        self.end = last
        self.mean = sum(array)/len(array)
        varsum = 0
        for int in array:
            varsum = varsum + (int-self.mean)*(int-self.mean)
        self.var = varsum/len(array)