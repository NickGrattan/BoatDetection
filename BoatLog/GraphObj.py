

class Point:
    def __init__(self, xcoord=0, ycoord=0):
        self.x = xcoord
        self.y = ycoord
    
    def dist(self, other):
        return math.hypot(abs(other.x - self.x), abs(other.y - self.y))

    def __str__(self):
        return "Point: x:{} y:{}".format(self.x,self.y)
        
class Rectangle:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def intersects_x(self, other):
        return (self.top_left.x <= other.bottom_right.x) and (other.top_left.x <= self.bottom_right.x)
    
    def intersects_y(self, other):
        return (self.top_left.y <= other.bottom_right.y) and (other.top_left.y <= self.bottom_right.y)
        
    def contains(self, other):
        return self.top_left.x < other.top_left.x < other.bottom_right.x < self.bottom_right.x and \
                    self.top_left.y < other.top_left.y < other.bottom_right.y < self.bottom_right.y
        
    def adjacent(self, other):
        adjacent_x = (abs(self.top_left.x - other.bottom_right.x) <= 1 or \
               abs(self.bottom_right.x - other.top_left.x) <= 1) and self.intersects_y(other)
        adjacent_y = (abs(self.top_left.y - other.bottom_right.y) <= 1 or \
               abs(self.bottom_right.y - other.top_left.y) <= 1) and self.intersects_x(other)
        return adjacent_x or adjacent_y
                
    def merge(self, other):
        tl = Point(min(self.top_left.x, other.top_left.x), min(self.top_left.y, other.top_left.y))
        br = Point(max(self.bottom_right.x, other.bottom_right.x), max(self.bottom_right.y, other.bottom_right.y))
        return Rectangle(tl, br)
    
    def __str__(self):
        return "Rectangle: top_left: {} bottom_right: {} ".format(self.top_left, self.bottom_right)
