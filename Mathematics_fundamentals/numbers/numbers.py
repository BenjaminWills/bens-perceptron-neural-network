import numpy as np
import scipy as sc


class Complex:
    def __init__(self, real, imaginary):
        self.re = real
        self.im = imaginary

    def __add__(self, other):
        if isinstance(other, Complex):
            return Complex(self.re + other.re, self.im + other.im)
        else:
            return complex(other + self.re, self.im)

    def __sub__(self, other):
        if isinstance(other, Complex):
            return Complex(self.re - other.re, self.im - other.im)
        else:
            return complex(self.re - other, self.im)

    def __mul__(self, other):
        if isinstance(other, Complex):
            return Complex(
                self.re * other.re - self.im * other.im,
                self.re * other.im + self.im * other.re,
            )
        else:
            return Complex(other * self.re, other * self.im)

    def get_conjugate(self):
        return Complex(self.re, -self.im)

    def get_complex_number(self):
        return f'{self.re}{"+" if self.im >= 0 else ""}{self.im}i'

    def get_magnitude(self):
        real = Real()
        return real.sqrt(self.re**2 + self.im**2)

    def get_argument(self):
        return np.arctan(self.im / self.re)


class Real:
    def sqrt(self, x):
        if x < 0:
            return Complex(0, np.sqrt(-x))
        else:
            return np.sqrt(x)

    @staticmethod
    def factorial(x):
        if x == 0:
            return 1
        else:
            return x * Real.factorial(x - 1)

    def get_powerset(self, arr: list):
        """
        This function will find the set of all subsets of an array,
        for an array of length n the power set has a length of 2 ** n.

        for [1,2] the power set is [],[1],[2],[1,2]. Note that it has 4 elements.
        """
        power_set = []
        n = len(arr)
        for i in range(2**n):
            sub_set = []
            for j in range(n):
                if i & 2**j:
                    sub_set.append(arr[j])
            power_set.append(sub_set)
        return power_set

    def is_prime(self, n):
        for i in range(2, int(n / 2)):
            if n % i == 0:
                return False
        return True

    def find_HD_and_LD(self, x):
        """
        Finds highest and lowest divisor of a number (one of which will always be prime.)
        """
        for i in range(2, int(x / 2)):
            if x % i == 0:
                LD = i
                HD = int(x / i)
                return [LD, HD]

    def prime_factors(self, n):
        i = n
        list_of_prime_factors = []
        while not self.is_prime(i):
            HD_LD = self.find_HD_and_LD(i)
            list_of_prime_factors.append(HD_LD[0])
            i = HD_LD[1]
        list_of_prime_factors.append(i)
        factor_dict = {}
        for i in list_of_prime_factors:
            if str(i) not in factor_dict.keys():
                factor_dict[str(i)] = 1
            else:
                factor_dict[str(i)] += 1
        return factor_dict

    @staticmethod
    def n_choose_m(n:int,m:int) -> int:
        """Will return the result of n choose m, i.e the number
        of ways to choose m objects from a group of n objects.

        Parameters
        ----------
        n : int
        m : int

        Returns
        -------
        int
            the number of ways to choose m objects from a group of n objects

        Raises
        ------
        error
            If m > n we throw an error.
        """
        if m > n:
            raise AttributeError(f'You cannot choose {m} items out of {n} items.')
        numerator = Real.factorial(n)
        denominator = Real.factorial(m) * Real.factorial(n-m)
        return numerator/denominator

    @staticmethod
    def gamma(z:float) -> float:
        """Returns the gamma function (z-1)!

        Parameters
        ----------
        z : float

        Returns
        -------
        float
            (z-1)!
        """
        return sc.special.gamma(z)
