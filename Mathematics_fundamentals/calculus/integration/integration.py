class Integration:
    @staticmethod
    def trapezium_approximation(function, start, end, steps):
        width = (start - end) / steps
        area = 0.5 * (function(start) + function(end))
        for i in range(1, steps - 1):
            area += function(start + i * width)
        return area * width

    @staticmethod
    def get_midpoint(p1, p2):
        return (p1 + p2) / 2

    @staticmethod
    def get_individual_area(function, p1, p2):
        midpoint = Integration.get_midpoint(p1, p2)
        funcval_at_start = function(p1)
        funcval_at_mid = function(midpoint)
        funcval_at_end = function(p2)
        return funcval_at_start + 4 * funcval_at_mid + funcval_at_end

    @staticmethod
    def simpson_approximation(function, start, end, steps):
        area = 0
        width = (end - start) / steps
        for i in range(steps):
            local_start = start + i * width
            local_end = start + (i + 1) * width
            area += Integration.get_individual_area(function, local_start, local_end)
        return width / 6 * area
