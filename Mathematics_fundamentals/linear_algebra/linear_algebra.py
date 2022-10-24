import numpy as np


class Matrix:

    """
    The Matrix object and associated functions.
    """

    def __init__(self,*args:list):
        self.matrix = []
        for row in args:
            self.matrix.append(row)
        self.rows = len(self.matrix)
        if len(args) > 0:
            self.columns = len(self.matrix[0])
        else:
            self.columns = 0

    def __add__(self,other):
        matrix_list = self.matrix
        other_matrix_list = other.matrix
        matrix = np.matrix(matrix_list)
        other_matrix = np.matrix(other_matrix_list)
        difference = matrix + other_matrix
        return self.numpy_to_matrix(difference)

    def __sub__(self,other):
        matrix_list = self.matrix
        other_matrix_list = other.matrix
        matrix = np.matrix(matrix_list)
        other_matrix = np.matrix(other_matrix_list)
        difference = matrix - other_matrix
        return self.numpy_to_matrix(difference)

    def __mul__(self,other):
        """
        Matrix multiplication works by multiplying rows by columns pairwise. Thus if mat1
        does not have the same number of rows as mat2 does columns, we cannot multiply them.

        Note: mat1 will be a list of lists to simulate a matrix:
            mat1 = [ [1,2,3]
                     [4,5,6]
                     [7,8,9]   ] 3x3 matrix.
        For example.

        In essence we can only multiply matrices with dimensions n x k and k x m, the resulting matrix
        will have dimension n x m. The plan is to transpose mat2 and then simply make the [i,j] element
        of the new matrix equal to the dot product of the i'th row of mat1 and the j'th column of mat2
        (which is the j'th row of mat2 transpose).
        """
        if isinstance(other,Matrix):
            matrix = Matrix()
            mat2_transpose = other.get_transpose()
            for row,row_vector in enumerate(self.matrix):
                row_to_add = []
                for column,column_vector in enumerate(mat2_transpose.matrix):
                    v1 = Vector(*row_vector)
                    v2 = Vector(*column_vector)
                    row_to_add.append(Vector.get_dot_product(v1,v2))
                matrix.add_rows(row_to_add)
            return matrix

        if isinstance(other,Vector):
            row_to_add = []
            for row,row_vector in enumerate(self.matrix):
                v1 = Vector(*row_vector)
                row_to_add.append(Vector.get_dot_product(v1,other))
            return Vector(*row_to_add)

        else:
            matrix = Matrix()
            for i in range(self.rows):
                row = []
                for j in range(self.columns):
                    row.append(other*self.matrix[i][j])
                matrix.add_rows(row)
            return matrix


    def show_matrix(self):
        """
        Gives visual representation of a matrix.
        """
        print('[')
        for row in self.matrix:
            print(str(row))
        print(']')

    def add_rows(self,*rows):
        """
        Adds rows to a matrix object.
        """
        for row in rows:
            self.matrix.append(row)
        # self.rows = len(self.matrix)
        return f"{len(rows)} rows added!"
    
    def add_columns(self,*columns):
        """
        Adds columns to a matrix object
        """
        # Case when matrix is empty
        if self.rows == 0:
            first_col = columns[0]
            for entry in first_col:
                self.matrix.append([entry])
            for index,column in enumerate(columns):
                if index > 0:
                    for index,value in enumerate(column):
                        self.matrix[index].append(value)
            self.rows = len(columns[0])
        # Otherwise.
        else:
            for column in columns:
                for index,value in enumerate(column):
                    self.matrix[index].append(value)                
        return f"{len(columns)} columns added!"

    def change_entry(self,row,column,new_value):
        """
        Will change an entry in a matrix
        """
        self.matrix[row][column] = new_value
        
        return f"Index ({row},{column}) changed to {new_value}"

    @staticmethod
    def get_empty_row(length):
        """
        returns an row vector of zeros.
        """
        row = Vector()
        zeros = [0]*length
        row.add_entries(*zeros)
        return Vector.unpack_vector(row)

    @staticmethod
    def get_empty_matrix(rows, columns):
        """
        Returns a rows x columns matrix with zeros for every entry.
        """
        matrix = Matrix()
        empty_rows = [Matrix.get_empty_row(columns)] * rows
        matrix.add_rows(*empty_rows)
        return matrix

    def get_transpose(self):
        """
        Will find the transpose of any matrix inputted. i.e will make the rows into columns and visa
        versa. The element at index [i,j] goes to [j,i] for all i in rows and j in columns.
        """
        transpose = Matrix()
        rows = self.matrix
        for row in rows:
            transpose.add_columns(row)
        return transpose
    
    def transform_function(self, x, function, matrix):
        """
        Will transform points of a function by using an inputted matrix transformation.
        """
        co_ordinate = Vector(x,function(x))
        new_co_ordinate = matrix*co_ordinate
        return Vector.unpack_vector(new_co_ordinate)

    def transform_conic_function(self, x, function, matrix):
        """
        One to many functions require special attention. This function will transform two points to new co ordinates
        as opposed to just one.
        """
        outputs = []
        for co_ordinates in function(x):
            co_ordinate = Vector(x, co_ordinates)
            new_co_ordinate = matrix*co_ordinate
            outputs.append(Vector.unpack_vector(new_co_ordinate))
        return outputs

    def get_determinant(self):
        """
        Gets determinant of an n x n matrix.
        """
        return np.linalg.det(self.matrix)

    @staticmethod
    def numpy_to_matrix(np_matrix):
        ben_matrix = Matrix()
        rows,columns = np_matrix.shape
        for row in range(rows):
            ben_matrix.add_rows(np_matrix[row,:])
        return ben_matrix

    def get_inverted_matrix(self):
        """
        Inverts n x n matrix that is non singular.
        """
        inv = Matrix.numpy_to_matrix(np.linalg.inv(self.matrix))
        return inv

    def get_eigenvalues(self):
        """
        Will get the eigenvalues of a matrix
        """
        return np.linalg.eigvals(self.matrix)[0]

    def get_eigenvectors(self):
        """
        Will get the eigenvectors of a matrix
        """
        return Matrix.numpy_to_matrix(np.linalg.eig(self.matrix)[1])

    def diagonalise_matrix(self):
        """
        Will diagonalise a matrix if that is possible (i.e if all eigenvalues are non degenerate.)
        """


    @staticmethod
    def solve_system_of_equations(matrix, vector):
        """
        Solving Ax = b, where x is a vector of unkowns, so x = inv(A)b.
        Only works when x has the same length as A and A is non singular.
        """
        if matrix.rows != len(vector.vector) or matrix.get_determinant() == 0:
            raise TypeError("Error! Invalid vector or singular matrix entered.")
        matrix_inverse = matrix.get_inverted_matrix()
        return matrix_inverse * vector

class Vector:

    def __init__(self,*args:int):
        self.vector = [[arg] for arg in args]
        self.dim = len(self.vector)

    def __add__(self,other):
        """
        Will add vectors component wise, only if they're the same shape.
        """
        v1 = self.vector
        v2 = other.vector
        dimv1 = len(v1)
        dimv2 = len(v2)
        packed_args = []
        if dimv1 == dimv2:
            for i in range(dimv1):
                packed_args.append(v1[i][0] + v2[i][0])
            return Vector(*packed_args)
        else:
            raise TypeError('Dimensions not equal.')

    def __sub__(self,other):
        """
        Will subtract vectors component wise, only if they're the same shape.
        """
        v1 = self.vector
        v2 = other.vector
        dimv1 = len(v1)
        dimv2 = len(v2)
        packed_args = []
        if dimv1 == dimv2:
            for i in range(dimv1):
                packed_args.append(v1[i][0] - v2[i][0])
            return Vector(*packed_args)
        else:
            raise TypeError('Dimensions not equal.')

    def __mul__(self,other):
        """
        Scalar multiplication for vectors, so other is a number.
        """
        if isinstance(other,Matrix):
            components = []
            for row in other.matrix:
                v = Vector(*row)
                components.append(Vector.get_dot_product(v,self))
            return Vector(*components)
        else:
            new_components = []
            for component in self.vector:
                new_components.append(1.0 * component[0] * other)
            return Vector(*new_components)

    def change_entry(self,new_entry,index):
        self.vector[index] = [new_entry]
        return f"Entry changed to {new_entry}"

    def add_entries(self,*new_entries):
        for entry in new_entries:
            self.vector.append([entry])
        return f"entries added!"

    def show_vector(self):
        """
        Visually showing a vector.
        """
        print('[')
        for index,component in enumerate(self.vector):
            if index == len(self.vector)-1:
                print(str(component))
            else:
                print(str(component) + ',')
        print(']')

    @staticmethod
    def get_dot_product(vector1, vector2) -> float:
        """
        Calculates the dot product between two vectors of equal length.
        """
        v1 = vector1.vector
        v2 = vector2.vector
        v_1_dim = len(v1)
        v_2_dim = len(v2)
        if v_1_dim != v_2_dim:
            raise TypeError(f"""Vector 1 has length {v_1_dim}, and vector 2 has length {v_2_dim}. 
            These must be equal.""")
        sum = 0
        for i in range(v_2_dim):
            sum += v1[i][0] * v2[i][0]
        return sum

    @staticmethod
    def unpack_vector(vector):
        """
        Will unpack a vector into its components, and then into an array.

        a = [
            [1],
            [2],
            [3]
        ]
        Vector.unpack(a) = [1,2,3]. This effectively transposes a vector!
        """
        unpacked_vector = []
        for i in vector.vector:
            unpacked_vector.append(i[0])
        return unpacked_vector

    @staticmethod
    def get_cross_product(vector1,vector2):
        """
        Will calculate the cross product between two vectors. These vectors must be three dimensional.
        note vectors must be instances of Vector.
        """
        v1,v2,v3 = Vector.unpack_vector(vector1)
        w1,w2,w3 = Vector.unpack_vector(vector2)

        cross_vector = [
            [v2*w3 - v3*w2],
            [v3*w1 - v1*w3],
            [v1*w2 - v2*w3]
        ]
        return cross_vector
    
    @staticmethod
    def get_matrix_from_vectors(*vectors):
        """
        Will make a matrix from vectors. Vectors go to columns.
        """
        matrix = Matrix()
        for vector in vectors:
            v_listified = Vector.unpack_vector(vector)
            matrix.add_columns(v_listified)
        return matrix

    def get_magnitude(self):
        return np.sqrt(Vector.get_dot_product(self,self))

    @staticmethod
    def get_unit_vector(position:int,dimension:int):
        """
        Will get a unit vector, i.e a vector that is all zeros bar one one.
        """
        zeroes = [0]*dimension
        basis_vector = Vector(*zeroes)
        basis_vector.change_entry(1,position)
        return basis_vector
