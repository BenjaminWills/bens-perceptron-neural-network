import random
from typing import Callable

import numpy as np
from scipy.stats import norm

from Mathematics_fundamentals.calculus.integration.integration import \
    Integration
from Mathematics_fundamentals.functions.functions import Functions
from Mathematics_fundamentals.numbers.numbers import Real


class Probability:
    """
    A class that contains all things probability!
    """
    

    @staticmethod
    def uniform_distribution_pdf(lower:float,upper:float) -> float:
        """_summary_

        Parameters
        ----------
        lower : float
        upper : float

        Returns
        -------
        float
            Probability of choosing x in the range [lower,upper]

        Raises
        ------
        ValueError
            If lower >= upper
        """

        if upper <= lower:
            raise ValueError("The upper bound is less than or equal to the lower bound.")
        return 1/(upper-lower)

    @staticmethod
    def uniform_distribution_cdf(lower:float,upper:float,x:float) -> float:
        """Will return the probability of finding a number <= x, within a uniform distribution
        defined on the range [lower,upper].

        Parameters
        ----------
        lower : float
            Lower bound
        upper : float
            Upper bound
        x : float
            The number that we are interested in, must be in range [lower,upper]

        Returns
        -------
        float
            probability that we find a number that is <= x.
        """
        if lower < upper:
            return random.uniform(lower,upper)
        else:
            raise ValueError('Lower is greater than or equal to upper.')

    @staticmethod
    def binomial_pdf(probability:float,n:int,x:int) -> float:
        """Will find the probability of finding exactly x occurances
        of an event in a binomial distribution

        Parameters
        ----------
        probability : float
            A number in (0,1] that defines the probability of success
        n : int
            Number of samples
        x : int
            Number of occurrences of event

        Returns
        -------
        float
            The probability of x events happening with a probability p of success 
            and n samples.
        """
        if n < x or probability < 0 or probability > 1:
            return TypeError("Invalid values entered.")
        n_choose_x = Real.n_choose_m(n,x)
        return n_choose_x * (probability ** x) * (1-probability) ** (n-x)

    @staticmethod
    def binomial_cdf(probability:float,n:int,x:int) -> float:
        """Will find the probability of finding <= x occurrences
        of an event in a binomial distribution

        Parameters
        ----------
        probability : float
            A number in (0,1] that defines the probability of success
        n : int
            Number of samples
        x : int
            Number of occurrences of event

        Returns
        -------
        float
            The probability of <= x events happening with a probability p of success 
            and n samples.
        """
        cdf = 0
        for i in range(x+1):
            cdf += Probability.binomial_pdf(
                probability = probability,
                n = n,
                x = i)
        return cdf

    @staticmethod
    def normal_pdf(x:float,mean:float,variance:float) -> float:
        """Returns the probability of choosing x in a normal distribution.

        Parameters
        ----------
        mean : float
            Mean value of normal distribution (horizontal shift)
        variance : float
            Variance of the normal distribution (stretch)
        x : float
            Value to evaluate probability of

        Returns
        -------
        float
            Probability of choosing exactly x in a normal distribution

        Raises
        ------
        ValueError
            Cannot have a negative variance.
        """
        if variance < 0:
            raise ValueError("Variance can't be negative")
        stretch_factor = 1/(np.sqrt(2 * np.pi * variance))
        exponential_factor = -1/2 * ((x-mean)**2)/variance
        return stretch_factor * Functions.exp(exponential_factor)

    @staticmethod
    def normal_cdf(mean:float,variance:float,x:float) -> float:
        """Returns the probability of choosing <= x in a normal distribution with mean and variance.

        Parameters
        ----------
        mean : float
            Mean value of normal distribution (horizontal shift)
        variance : float
            Variance of the normal distribution (stretch)
        x : float
            Value to evaluate probability of

        Returns
        -------
        float
            The probability of choosing <= x in a normal distribution with mean and variance.
        """
        integratable_pdf = lambda x : Probability.normal_pdf(x,mean,variance)
        return Integration.simpson_approximation(
            function = integratable_pdf,
            start = -10_000,
            end = x,
            steps = 10000
        )

    @staticmethod
    def standard_normal_cdf(mean:float,variance:float,x:float) -> float:
        if mean !=0 and variance != 1:
            standardised_import = (x-mean)/np.sqrt(variance)
        return Probability.normal_cdf(
            mean=0,
            variance=1,
            x = standardised_import
        )

    @staticmethod
    def inverse_normal(mean:float,variance:float,p:float) -> float:
        """ If we have some probability p in the range [0,1] we can find
        the corresponding value, x such that the probability that the normal
        variable is <= x is p.

        Parameters
        ----------
        mean : float
        variance : float
        p : float
            Probability in the range [0,1]

        Returns
        -------
        float
            Corresponding value to the probability
        """
        return np.sqrt(variance) * norm.ppf(p) + mean
    
    @staticmethod 
    def geometric_pdf(probability:float,x:int) -> float:
        """The geometric pdf, the probability that you fail k times and then
        have a success

        Parameters
        ----------
        probability : float
            probability of success
        x : int
            Number of failures before a success

        Returns
        -------
        float
            The probability of having x failures before a success

        Raises
        ------
        ValueError
            Probability must be in range [0,1] and x has to be > 0
        """
        if probability < 0 or x < 0 or probability > 1:
            raise ValueError()
        return probability * (1-probability) ** x

    @staticmethod
    def geometric_cdf(probability:float,x:int) -> float:
        """The geometric cdf, the probability that you fail at most k times and then
        have a success

        Parameters
        ----------
        probability : float
            probability of success
        x : int
            Number of failures before a success

        Returns
        -------
        float
            The probability of having at most x failures before a success

        Raises
        ------
        ValueError
            Probability must be in range [0,1] and x has to be > 0
        """
        cdf = 0
        for i in range(x+1):
            cdf += Probability.geometric_pdf(
                probability=probability,
                x = i
                )
        return cdf

    @staticmethod
    def poisson_pdf(rate:float,x:int) -> float:
        """Will find the poisson pdf, which is the limiting case of the binomial distribution
        in which the number of samples, n approaches infinite.

        Parameters
        ----------
        rate : float
            The rate (number of events / period)
        x : int
            x events in a certain time period

        Returns
        -------
        float
            probability of seeing x events

        Raises
        ------
        ValueError
            Rate can only be > 0.
        """
        if rate <= 0:
            raise ValueError('Rate < 0')
        return (rate ** x * Functions.exp(-rate))/Real.factorial(x)

    @staticmethod
    def poisson_cdf(rate:float,x:int) -> float:
        """Will find the poisson cdf, which is the limiting case of the binomial distribution
        in which the number of samples, n approaches infinite.

        Parameters
        ----------
        rate : float
            The rate (number of events / period)
        x : int
            x events in a certain time period

        Returns
        -------
        float
            probability of seeing at most x events in a time period

        Raises
        ------
        ValueError
            Rate can only be > 0.
        """
        cdf = 0
        for i in range(x+1):
            cdf += Probability.poisson_pdf(
                rate = rate,
                x = i
            )
        return cdf

    def exponential_pdf(rate:float,x:float) -> float:
        """The exponential pdf, 

        Parameters
        ----------
        rate : float
            Rate of occurrence of event
        x : float
            Duration (usually)

        Returns
        -------
        float
            Probability that something happens in this event

        Raises
        ------
        ValueError
            Negative rate
        """
        if rate <= 0:
            raise ValueError('Rate must be > 0')
        if x <= 0:
            return 0
        return rate * Functions.exp(- rate * x)

    def exponential_cdf(rate:float,x:float) -> float:
        """The exponential cdf, 

        Parameters
        ----------
        rate : float
            Rate of occurrence of event
        x : float
            Duration (usually)

        Returns
        -------
        float
            Probability that something happens in this event

        Raises
        ------
        ValueError
            Negative rate
        """
        return 1 - Functions.exp(- rate * x)

    @staticmethod
    def gamma_pdf(n:int,rate:float,x:float) -> float:
        """Returns gamma pdf, the sum of n exponential variables
        with rate: rate. I.e the amount of time taken for the k'th event to occur

        Parameters
        ----------
        n : int
            Number of occurrences
        rate : float
            Rate of event occurrence
        x : float
            Time taken for n occurrences to occur

        Returns
        -------
        float
            probability of n events occurring in x time

        Raises
        ------
        ValueError
            n,rate should be > 0
        """
        if n < 0 or rate < 0:
            raise ValueError()
        return ((rate ** n)/Real.gamma(n)) * (x ** (n-1)) * Functions.exp(-rate*x)
    
    @staticmethod 
    def gamma_cdf(n:int,rate:float,x:float) -> float:
        """Returns gamma cdf, the sum of n exponential variables
        with rate: rate. I.e the amount of time taken for the k'th event to occur

        Parameters
        ----------
        n : int
            Number of occurrences
        rate : float
            Rate of event occurrence
        x : float
            Time taken for n occurrences to occur

        Returns
        -------
        float
            probability of n events occurring in <= x time

        Raises
        ------
        ValueError
            n,rate should be > 0
        """
        integrand = lambda z: Probability.gamma_pdf(
            n = n,
            rate = rate,
            x = z
        )
        return Integration.simpson_approximation(
            function = integrand,
            start = -10_000,
            end = x,
            steps = 100_000
        )

class Empirical_probability:
    @staticmethod
    def get_mean(data:list) -> float:
        """Will find the mean of a list of data

        Parameters
        ----------
        data : list
            A list of numbers

        Returns
        -------
        float
            The mean of the inputted list
        """
        n = len(data)
        sum = sum(data)
        return sum/n

    @staticmethod
    def get_variance(data:list) -> float:
        """The variance is the average square distance of data points 
        from the mean

        Parameters
        ----------
        data : list
            A list of numbers

        Returns
        -------
        float
            Variance
        """
        variance = 0
        n = len(data)
        mean = Empirical_probability.get_mean(data)
        for data_point in data:
            variance += (mean - data_point) ** 2
        return variance / n

class Sampling:
    def get_sample(N:int,inverted_distribution:Callable) -> list:
        """Will get a sample of size N using the uniform CDF method.
        i.e we generate a sample based on the inverse of the CDF in
        question.

        Parameters
        ----------
        N : int
            Number of samples
        inverted_distribution : Callable
            The inverse of the distribution you need to sample from,
            hint: use the invert_scalar function from Functions.

        Returns
        -------
        list
            A sample that follows the distribution.
        """
        sample = []
        count = 1
        while count < N:
            random_probability = np.random.uniform(0,1)
            output = inverted_distribution(random_probability)
            sample.append(output)
            count += 1
        return sample