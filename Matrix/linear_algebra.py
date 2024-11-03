# Modules for data classes and manipulations
from random import randint, random
from math import gamma, prod, log2, ceil
# Modules for functions and iterables
from collections import deque
from itertools import product, chain, starmap, pairwise, accumulate, count
from more_itertools import padded, chunked, all_equal
from functools import cached_property, partial, reduce
from operator import *
from copy import *
# Modules for type annotation & determination
from numbers import Number
from typing import *
from types import *
from collections.abc import *
# Modules for window input
import tkinter as tk
import ctypes


def _auto_type(num):
    num = float(num)
    return num if num % 1 else int(num)


def _is_matrix(iterable):
    # The function is written to recognize the data structure within a iterable with the form of matrix.
    # This is done by check if all of the iterables within are of the same length.
    try             : return all_equal(iterable, key=len)
    except TypeError: return False


def _dot(vec1, vec2):
    # Dot product between two vectors
    return sum(map(mul, vec1, vec2))


class _TableInput:
    def __init__(self, rows, columns, dtype=None):        
        self.dtype = _auto_type if dtype is None else dtype
        self.rows = rows
        self.columns = columns
        self.table = []
        self.current_row = 0
        self.current_column = 0
        
        self.root = tk.Tk()
        self.root.title("Matrix Input")
        
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)
        
        self.entry_widgets = []
        for row in range(rows):
            row_widgets = []
            for column in range(columns):
                entry = tk.Entry(self.frame, width=10)
                entry.grid(row=row, column=column, padx=5, pady=5)
                row_widgets.append(entry)
            self.entry_widgets.append(row_widgets)
        
        self.submit_button = tk.Button(self.frame, text="Submit", command=self.submit_table)
        self.submit_button.grid(row=rows+1, column=0, padx=5, pady=5)
        
        self.root.bind("<Return>", self.submit_table)
        self.root.bind("<Right>", self.move_right)
        self.root.bind("<Left>", self.move_left)
        self.root.bind("<Up>", self.move_up)
        self.root.bind("<Down>", self.move_down)
        self.root.bind("<Escape>", self.close_without_input)

        self.root.attributes('-topmost', True)
        self.root.after(0, self.get_focus)

    def get_focus(self):
        ctypes.windll.user32.keybd_event(0x12, 0, 0x0001 | 0, 0)
        ctypes.windll.user32.SetForegroundWindow(self.root.winfo_id())
        ctypes.windll.user32.keybd_event(0x12, 0, 0x0001 | 0x0002, 0)
        self.entry_widgets[0][0].focus_set()

    def move_right(self, event):
        if self.current_column < self.columns - 1:
            self.current_column += 1
        else:
            self.current_column = 0
            if self.current_row < self.rows - 1:
                self.current_row += 1
            else:
                self.current_row = 0
        self.entry_widgets[self.current_row][self.current_column].focus_set()

    def move_left(self, event):
        if self.current_column > 0:
            self.current_column -= 1
        else:
            self.current_column = self.columns - 1
            if self.current_row > 0:
                self.current_row -= 1
            else:
                self.current_row = self.rows - 1
        self.entry_widgets[self.current_row][self.current_column].focus_set()

    def move_up(self, event):
        if self.current_row > 0: self.current_row -= 1
        self.entry_widgets[self.current_row][self.current_column].focus_set()

    def move_down(self, event):
        if self.current_row < self.rows - 1: self.current_row += 1
        self.entry_widgets[self.current_row][self.current_column].focus_set()

    def submit_table(self, event=None):
        for row in self.entry_widgets:
            row_data = []
            for entry in row:
                if entry.get() != "":
                    row_data.append(self.dtype(entry.get()))
            if row_data:
                self.table.append(row_data)
        self.root.destroy()

    def close_without_input(self, event=None):
        self.root.destroy()
        raise SystemExit

    def run(self):
        self.root.mainloop()
        return self.table


class matrix(tuple):
    """The core built-in class in this package: matrix."""
    cache = True

    ## Matrix build
    
    def __new__(cls, *args, **kwargs):
        """        
        There are 3 ways to build a matrix:

        Direct build
            Use a 2-dimensional list that has the form of matrix.
            e.g.:
            > matrix([[1, 2], [3, 4]])
            The lengths of row vectors must be all equal,
            otherwise they will be recognized as entries.
                     
        Joint build
            Build a matrix using several lists that only contain values and whose lengths are all equal.
            This is similar with the way of using several row or column vectors to construct a matrix.
            e.g.:
            >>> matrix([1, 2], [3, 4])
            ╭1  2╮
            ╰3  4╯
            >>> matrix([1, 2], [3, 4], as_column=True)
            ╭1  3╮
            ╰2  4╯

        Specification build
            Build a matrix in a way specified by arguments using input data.
            Allowed arguments include:
                rn        : Number of rows. (Divide the numbers into rn sub-lists.)
                cn        : Number of columns. (The length of each sub-list.)
                fill      : The value with which the vacancies are filled in.
                            Vacancies occur when the number of values typed in are fewer than entries specified.
                vertical  : Arrange the numbers column-wise, so that every rn numbers are put into one column. 
                horizontal: Arrange the numbers row-wise, so that every cn numbers are put into a row.
                            This is the default orientation of arrangement.
                as_column : The same as "vertical".
                as_row    : The same as "horizontal".
                reverse   : Place the numbers reversedly.
                as_diag   : Arange the data at the diagonal positions.
                repeat    : Repeat some entries or row vectors.

        Note:
            1. There are by principle no restrictions on the type of data included in a matrix,
               But most of the calculations cannot apply to data that are not numbers.

            2. The matrix class is a subclass of list, and it inherits all of its method.

            3. The type of the sub-list is recommended to be tuple or list, but is free of choice.
        """
        L = len(args)
        proto = ((),) if L == 0 else args[0]
        # proto may be a generator resulted from expression comprehension
        # regenerate if this condition occurs
        try: proto[0]
        except TypeError:
            if isinstance(proto, Iterable):
                proto: Union[Generator, Iterator] # Iterable but unsubscriptable
                proto = tuple(tuple(row) for row in proto)
        except IndexError: proto = ((),) # proto is empty list/tuple
        ## Direct build.
        if _is_matrix(proto):
            proto: Sequence[Sequence] # nested tuple/list, just like the matrix format
            if L > 1:    raise TypeError(f'expected 1 argument, got {L}')
            elif kwargs: raise TypeError("direct build does not accept keyword arguments")
        ## Joint build.
        elif _is_matrix(args):
            args: tuple[Sequence]
            proto = tuple(zip(*args)) if kwargs.get('as_column') else args
        ## Specification build.
        else:
            args: tuple[Number]
            # The value of "fill" is deduced from data type input.
            # This argument is not always viable because some classes do not allow to be empty.
            fill = kwargs.get('fill') if 'fill' in kwargs else type(args[0])()
            if (rp := kwargs.get('repeat')): args, L = args*rp, L*rp # repeat a value or a vector
            if kwargs.get('as_diag'):
                rn = kwargs['rn'] if 'rn' in kwargs else L
                cn = kwargs['cn'] if 'cn' in kwargs else L # if no shape info is specified, build a square diagonal matrix
                num_diag = min((rn, cn))
                if (l := len(args)) < num_diag: args += (fill, ) * (num_diag - l)
                proto = ()
                for i in range(num_diag - 1): proto += (args[i],) + (0, )*cn
                proto += (args[num_diag-1],) + (0, ) * (rn*cn - (cn+1)*(num_diag-1) - 1)
                args=proto
            else:
                if 'rn' in kwargs:
                    rn = kwargs['rn']
                    cn = kwargs['cn'] if 'cn' in kwargs else L//rn # this probably discards several values at the end
                else:
                    cn = kwargs['cn'] if 'cn' in kwargs else L     # if no shape info is specified, build a row vector
                    rn = L//cn
                if (l := rn*cn ) > L: args += (fill, )*(l - L)
            if kwargs.get('reverse'): args = args[::-1]           # reverse the order of data
            if kwargs.get('vertical') or kwargs.get('as_column'): # transpose
                  proto = tuple(zip(*chunked(args, rn)))
                  if kwargs.get('horizontal') or kwargs.get('as_row'):
                      raise TypeError('conflicting arguments in matrix building')
            else: proto = tuple(chunked(args, cn))
                
        self = super().__new__(cls, proto)

        self.__proto = proto
        self.shape   = self.rn, self.cn = len(proto), len(proto[0])
        self.square  = (self.rn == self.cn)
        return self

    @classmethod
    def input(cls, n_row=10, n_col=10, dtype=None):
        """Type a matrix in a table window."""
        n_row: int
        n_col: int
        dtype: Union[type, FunctionType, LambdaType]
        
        DataInput = _TableInput(n_row, n_col, dtype)
        data      = DataInput.run()
        return cls(data)

    @classmethod
    def from_csv(cls, csv, dtype=None):
        if dtype is None: dtype = _auto_type
        with open(csv, 'r', encoding='utf-8') as f:
            return cls([[dtype(value) for value in line.split(",")] for line in f.readlines()])

    @classmethod
    def from_str(cls, string, dtype=None):
        if dtype is None: dtype = _auto_type
        lines = [[dtype(num) for num in line.strip("╭ ╮│╰╯()").split()] for line in string.split('\n')]
        return cls(lines)
        
    
    def __eq__(self, other):
        if isinstance(other, matrix):
            return self.shape == self.shape and all(map(eq, self.entry, other.entry))
        else:
            return self.to_tuple() == other
    
    ## Matrix statistics

    def __contain__(self, query):
        """Test whether a value is an entry of the matrix."""
        for row in self.__proto:
            if query in row:
                return True
        else:   return False

    def count_entry(self, entry):
        """Count the occurence of an entry in the matrix."""
        return sum(map(lambda x: x.count(entry), self.__proto))

    def is_block_of(self, other):
        assert self.rn <= other.rn and self.cn <= other.cn
        
        def find_slice(lst1, length1, lst2, length2):
            locs = deque()
            for i in range(length2 - length1 + 1):
                if tuple(lst2[i: i+length1]) == tuple(lst1): locs.append(i)
                else: continue
            return locs

        x1 = self.__proto[0]
        for i in range(other.rn - self.rn + 1):
            locs = find_slice(x1, self.cn, other.__proto[i], other.cn)
            for loc in locs:
                for j in range(1, self.rn):
                    if self.__proto[j] == other.__proto[i+j][loc: loc+self.cn]:
                        continue
                    else:
                        break
                else:
                    return True
        else:
            return False

    def count_row(self, row):
        return self.count(row)

    def count_col(self, col):
        return self.t.count(col)

    @property
    def order(self):
        assert self.square
        return self.rn

    def __len__(self):
        """The number of entries in a matrix."""
        return self.cn * self.rn

    @property
    # The all of entries in the matrix as a list can be accessed as an iterator.
    def entry(self):
        return chain.from_iterable(self.__proto) 

    def flatten(self):
        return tuple(self.entry)

    def to_list(self):
        return [list(item) for item in self]

    def to_tuple(self):
        return tuple(tuple(item) for item in self)

    def to_iterator(self):
        return iter(iter(item) for item in self)

    def to_deque(self):
        return deque(deque(item) for item in self)

    def __bool__(self):
        """The boolean value of a empty matrix is False, else it's True."""
        return len(self.__proto[0]) != 0

    ## Matrix presentation and identity

    def __str__(self):
        if self.rn == 0:
            return "()"
        else:
            pro = [[str(self.__proto[i][j]) for j in range(self.cn)] for i in range(self.rn)]
            if self.rn == 1:
                matrix_form = "{}  "*self.cn
                return "( " + matrix_form.format(*pro[0]).strip() + " )"
            else:
                lengths = tuple(max([len(pro[i][j]) for i in range(self.rn)]) for j in range(self.cn))
                matrix_form = ""
                for i in range(self.rn):
                    row = ("{:>%d}  "*(self.cn - 1) + "{:>%d}")%lengths
                    if i == 0:
                        matrix_form += "╭ " + row.format(*pro[i]) + " ╮" + "\n"
                    elif 0 < i < self.rn - 1:
                        matrix_form += "│ " + row.format(*pro[i]) + " │" + "\n"
                    else:
                        matrix_form += "╰ " + row.format(*pro[i]) + " ╯"
                return matrix_form

    def __repr__(self):
        if matrix.cache:
            try:
                list(globals().keys())[list(globals().values()).index(self)]
            except ValueError:
                global _cached_matrix
                _cached_matrix = self
        return str(self)

    @classmethod
    def last(cls):
        try:
            last = globals()['_cached_matrix']
            global _cached_matrix
            del _cached_matrix
            return last
        except KeyError:
            raise MemoryError("there is no matrix cached")

    def __deepcopy__(self):
        return matrix(deepcopy(self.__proto))

    def __copy__(self):
        return self.__deepcopy__()

    ## Matrix opertation

    def __arithmetic_operation(self, other, op):
        op: FunctionType
        if isinstance(other, matrix):
            assert self.shape == other.shape
            # Flatten the matrix first into a single-level iterator then execute the operation,
            # after which the iterator is divided again into a matrix.
            proto = chunked(map(op, self.entry, other.entry), self.cn)
            return matrix(tuple(proto))
        else:
            opx = lambda x: op(x, other)
            proto = chunked(map(opx, self.entry), self.cn)
            return matrix(tuple(proto))

    def __add__(self, other): return self.__arithmetic_operation(other, op=add) # "+" operator
    def __sub__(self, other): return self.__arithmetic_operation(other, op=sub) # "-" operator
    def __mod__(self, other): return self.__arithmetic_operation(other, op=mod) # "%" operator
    def __truediv__(self, other):  return self.__arithmetic_operation(other, op=truediv)  # "/" operator
    def __floordiv__(self, other): return self.__arithmetic_operation(other, op=floordiv) # "//" operator

    def __radd__(self, other): return self + other # In case a number is in the front of the matrix

    # Multiplication of matrices
    def __mul__(self, other):
        """
        The multiplication of matrix: "*" operator.
        Multiplication by number: Multiply every entry from the matrix by the number.
        Multiplication between matrices:
        Standard multiplication: the entry at i-th row and j-th column is the dot-product between the i-th row from the former matrix and the j-th column of the latter matrix.
        Multiplication by number: the multiplication between matrices may not satisfy commutative law, which is that AB may not be equal to BA.
        Hadamard product: multiply the entries at corresponding positions from two matrices.
        """
        if isinstance(other, matrix) and self.cn == other.rn:
            # Transpose the other matrix first, and calculate row vector dot-multiply row-vector.
            # It is rather faster to tranpose first and fetch the row than accessing every column of the original 'other' matrix.
            proto = chunked(starmap(_dot, product(self.__proto, zip(*other.__proto))), other.cn)
            return matrix(tuple(proto))
        else:
            return self.__arithmetic_operation(other, op=mul)

    def __matmul__(self, other):
        "The outer product/Kronecker product/tensor product of matrices."
        return matrix(*map(lambda y: tuple(starmap(mul, y)), map(lambda x: product(*x), product(self, other))))

    def kronecker_sum(self, other):
        return self @ E(other.order) + other @ E(self.other)

    def hadamard(self, other):
        assert isinstance(other, matrix)
        return self.__arithmetic_operation(other, op=mul)

    def slyusar(self, other):
        """
        Face-splitting product.
        Firstly proposed by Vadym. I. Slyusar (1964-).
        Reference:
        Slyusar, V. I. (December 27, 1996). "End matrix products in radar applications" . Izvestiya VUZ: Radioelektronika. 41 (3): 71–75.
        Slyusar, V. I. (1997-05-20). "Analytical model of the digital antenna array on a basis of face-splitting matrix products" . Proc. ICATT-97, Kyiv: 108–109.
        Slyusar, V. I. (1997-09-15). "New operations of matrices product for applications of radars" . Proc. Direct and Inverse Problems of Electromagnetic and Acoustic Wave Theory (DIPED-97), Lviv.: 73–74.
        """
        assert self.rn == other.rn
        return matrix(*map(lambda x: tuple(starmap(mul, x)), map(product, self, other)))

    def kronecker_col(self, other):
        """
        Column-wise Kronecker product.
        It is in fact the transpose version of the face-splitting product.
        """
        assert self.cn == other.cn
        self_t, other_t = zip(*self.__proto), zip(*other.__proto)
        return matrix(*zip(*map(lambda x: tuple(starmap(mul, x)), map(product, self_t, other_t))))

    def khatri_rao(self, other, part_a, part_b):
        """
        Khatri-Rao product of matrices:
        A kind of mixed product of matrices.
        The calculation process is:
        Firstly partition the matrix, assure that the number of row and column blocks are same, no matter what their shapes are.
        Then, pair the blocks like Hadamard product and calculate the Kronecker product for each block pair.
        Example:
        >>> a=matrix(*range(1, 10), rn=3)
        >>> b=matrix(*range(1, 10), rn=3, as_column=1)
        >>> a.khatri_rao(b, [[2, 1], [2, 1]], [[1, 2], [1, 2]])
        
        ╭  1   2  12  21 ╮
        │  4   5  24  42 │
        │ 14  16  45  72 │
        ╰ 21  24  54  81 ╯
        """
        part_a: list[list[int], list[int]]
        part_b: list[list[int], list[int]]
        assert len(part_a[0]) == len(part_b[0]) and \
               len(part_a[1]) == len(part_b[1])
        a_parted = self.partition(*part_a)
        b_parted = other.partition(*part_b)
        res_init = [[a_parted[i][j] @ b_parted[i][j] for j in range(len(part_a[1]))] for i in range(len(part_a[0]))]
        return matrix.block_union(res_init)

    def tracy_singh(self, other, part_a, part_b):
        """
        Tracy-Singh product.
        The partition of each matrix is arbitrary.
        Pair the blocks like Kronecker product, and compute the Kronecker product for each pair.
        """
        part_a: list[list[int], list[int]]
        part_b: list[list[int], list[int]]
        a_parted = self.partition(*part_a)
        b_parted = other.partition(*part_b)
        res_parted = list(map(lambda y: tuple(starmap(matmul, y)), map(lambda x: product(*x), product(a_parted, b_parted))))
        return matrix.block_union(res_parted)

    kronecker = __matmul__
    face_splitting = slyusar

    def __rmul__(self, other):
        """Multiplication of matrix by a number."""
        mulx  = lambda x: x*other
        proto = chunked(map(mulx, self.entry), self.cn)
        return matrix([*proto])
    
    def __pow__(self, other):
        """
        Calculate the power of matrix: "**" operator.
        Only square matrix can be multiplied by itself.
        This method now only supports the matrix to the power of an integer. 
        """
        assert other%1 == 0
        _mul = lambda a, b: chunked(starmap(_dot, product(a, zip(*b))), self.rn)
        power = [[1 if i==j else 0 for i in range(self.order)] for j in range(self.order)]
        if other == 0: return matrix(*power) # Zero-th power is the identity matrix
        elif other > 0: # Positive power is the power of self
            for time in range(other): power = _mul(power, self.__proto)
            return matrix(*power)
        else:
            inv = self.inv.__proto
            for time in range(-other): power = _mul(power, inv)
            return matrix(*power)
            # Negative power is the power of inverse
            # The matrix must be invertable, i.e. the determinant must not be zero.

    def p(self, *args, shift=0):
        """
        The polynomial of the matrix. For args a0, a1, a2, ..., an calculate:
        a0*E + a1*Aij + a2*Aij**2 + ... + an*Aij**n.
        Only sqaure matrces have polynomials.
        """
        args: tuple[Number]; shift: int
        sum_it = 0
        for i in range(len(args)):
            sum_it += args[i]*(self**(i-shift))
        return sum_it

    def __neg__(self):
        """
        The negative of a matrix: "-" operator with no minuend.
        Multiply the matrix by (-1).        
        """
        proto = chunked(map(neg, chain.from_iterable(self.__proto)), self.cn)
        return matrix(tuple(proto))

    ## Matrix manipulation
    
    def __getitem__(self, item):
        """
        Matrix subscription:
        Tuple index is implemented here, which is similar to the natural matrix mathematics.
        Ellipsis(...) can be used to refer to all entries in a context that is necessary.
        For e.g.
        >>> mtr[0]  # row vector
        >>> mtr[1,] # column vector, which is equivalent to mtr[..., 1]
        >>> mtr[0, 1] # Access a single value in the matrix, which is equivalent to mtr[0][1] or mtr[1,][0]
        """
        item: Union[EllipsisType, int, slice, tuple[int|slice]]
        if item is Ellipsis:   return self
        elif isinstance(item, int | slice):
            proto = self.__proto
            if self.rn == 1:   return proto[0][item]
            elif self.cn == 1: return proto[item][0]
            else:              return matrix(self.__proto[item])
        elif isinstance(item, tuple):
            if len(item) == 0: return self
            elif len(item) == 1:
                item = item[0]
                if item is Ellipsis:          return self
                elif isinstance(item, slice): return matrix(*map(lambda x: x[item], self.__proto))
                else:                         return matrix(*map(lambda x: x[item], self.__proto), as_column=True)
            elif len(item) == 2:
                submtr = self[item[0]][item[1],]
                return submtr if len(submtr) > 1 else submtr.__proto[0][0]
            else: raise IndexError("Invalid indices")

    def row(self, i):
        """Get a row of the matrix, using row number insread of index."""
        return matrix(self.__proto[i-1])

    def col(self, j):
        """Get a column of the matrix."""
        return matrix(tuple(item[j-1: j] for item in self.__proto))

    def partition_by_index(self, row=None, col=None):
        """
        Partition a matrix using the indices of rows/columns at the border between sub-matrices.
        That is, one must specify the position (index) of each row/column, at which the matrix is divided.
        The returned will be a list of sub-matrices, nested into 2 levels, exactly like the data structure of matrix.
        """
        row: Optional[Sequence[int, int]];
        col: Optional[Sequence[int, int]]
        
        if row is None: row = []
        if col is None: col = []
        row, col = chain([0], row, [self.rn]), chain([0], col, [self.cn])
        # Generate the slices used for partitioning
        rindex = pairwise(row)
        cindex = tuple(pairwise(col))
        parted_row = [self.__proto[i[0]: i[1]] for i in rindex]
        # The final result is a nested list of matrices.
        parted = [matrixList([matrix(tuple(row[j[0]:j[1]] for row in mtr)) for j in cindex]) for mtr in parted_row]
        return parted

    def partition_into(self, rows=2, cols=2):
        """Partition a matrix into a number of rows and columns."""
        def _divide(n, iterable):
            seq = tuple(iterable)
            q, r = divmod(len(seq), n)
            stop = 0
            for i in range(1, n + 1):
                start = stop
                stop += q + 1 if i <= r else q
                yield seq[start: stop]
        row_parted = _divide(rows, self)
        parted = [matrixList([matrix(item) for item in zip(*[_divide(cols, row) for row in mtr])]) for mtr in row_parted]
        return parted

    def partition(self, rns=None, cns=None):
        """Partition a matrix using the number of rows/columns of each sub-matrices."""
        row: Optional[Sequence[int]] ;col: Optional[Sequence[int]]
        if rns is None: rns = [self.rn]
        if cns is None: cns = [self.cn]
        assert sum(rns) == self.rn and sum(cns) == self.cn
        row, col = tuple(accumulate(rns))[:-1], tuple(accumulate(cns))[:-1]
        return self.partition_by_index(row, col)

    block = partition

    @classmethod
    def block_union(cls, matrix_of_matrices):
        matrix_of_matrices: list[list[matrix]]
        # Assert the validity of blocks
        for row in matrix_of_matrices:
            row: list[matrix]
            assert all_equal(row, lambda x: x.rn)
        for col in zip(*matrix_of_matrices):
            assert all_equal(col, lambda x: x.cn)
        # Implement the merge
        for i in range(len(matrix_of_matrices)):
            matrix_of_matrices[i] = reduce(or_, matrix_of_matrices[i])
            matrix_of_matrices[i]: matrix
        matrix_of_matrices = reduce(and_, matrix_of_matrices)
        return matrix_of_matrices

    def __and__(self, other):
        """Join two matrices by row: "&" operator."""
        assert isinstance(other, matrix) and self.cn == other.cn
        return matrix(self.__proto + other.__proto)

    def __or__(self, other):
        """Join two matrices by column: "|" operator."""
        assert isinstance(other, matrix) and self.rn == other.rn
        return matrix(*map(add, self.__proto, other.__proto))

    def __lshift__(self, other):
        """
        The translation of matrix: "<<" opertator (leftward or upward).
        If a integer is given, translate row-wise (vertically);
        If a tuple is given, translate column-wise (the tuple has only one entry) or row-wise and column-wise (the tuple has 2 entries).
        """
        other: int
        left, upper = (other, 0) if isinstance(other, int) else ((0,)*(2-len(other)) + other)
        left, upper = (left%self.cn, upper%self.rn)
        proto = self.__proto[upper:] + self.__proto[:upper]
        proto = map(lambda x: x[left:] + x[:left], proto)
        return matrix(*proto)

    def __rshift__(self, other):
        other: int
        """The translation of matrix: ">>" opertator (rightward or downward)."""
        right, lower = (other, 0) if isinstance(other, int) else ((0,)*(2-len(other)) + other)
        right, lower = (right%self.cn, lower%self.rn)
        proto = self.__proto[-lower:] + self.__proto[:-lower]
        proto = map(lambda x: x[-right:] + x[:-right], proto)
        return matrix(*proto)

    def __reversed__(self):
        """The reversed() method: reverse rows."""
        return matrix(list(reversed(self.__proto)))

    def __invert__(self):
        """The "~" operator: reverse columns."""
        proto = map(lambda x: list(reversed(x)), self.__proto)
        return matrix(*proto)

    @property
    def tr(self):
        """The trace of a square matrix, which is the sum of diagonal entries."""
        return sum(self.__proto[x][x] for x in range(self.order))

    def r_sum(self, r):
        """The sum of a row."""
        return sum(self.__proto[r])

    def c_sum(self, c):
        """The sum of a column."""
        return sum(self.__proto[i][c] for i in range(self.rn))

    @cached_property
    def d(self):
        """
        The determinant of a square matrix:
        Implementation by linear transformation algorithm.
        """
        assert self.square
        lst  = self.to_list()
        flag, det = 1, 1

        if len(lst) == 1: return lst[0][0]

        # Swap the first row with another row whose first entry is not equal to 0
        # Because of the swap, the value of the determinant is the negative of the original.
        # If the all entries from a row is all 0, then the value of determinant is 0.
        # This operation costs little time compared to the computation of the determinant itself.
        for COUNT in range(len(lst)-2):
            x1 = lst[0]
            if x1[0] == 0:
                for xi in lst[1:]:
                    if xi[0] != 0:
                        xi[:], x1[:] = x1[:], xi[:]
                        flag = - flag
                        break
                else: return 0

            # Implement the Gauss-Jordan elimination.
            for xi in lst[1:]:
                if xi[0] != 0:
                    R = xi[0]/x1[0]
                    xi[:] = [xij - R*x1j for xij, x1j in zip(xi[1:], x1[1:])]
                else:
                    xi[:] = xi[1:]
            lst[:] = lst[1:]
            det *= flag * x1[0]
        return det * (lst[0][0] * lst[1][1] - lst[0][1] * lst[1][0])

    @cached_property
    def D(self): return self.d

    def minor(self, i, j):
        """
        Get the minor matrix M_ij of a matrix,
        in which the i-th row and j-th column are not present.
        """
        proto = self.__proto[:i-1] + self.__proto[i:]
        return matrix(item[:j-1] + item[j:] for item in proto)

    def cofactor(self, i, j):
        """The cofactor of the matrix."""
        return self.minor(i, j).d

    @cached_property
    def adj(self):
        """The adjugate matrix of the current matrix by definition."""
        return matrix([[self.cofactor(i+1, j+1) for j in range(self.cn)] for i in range(self.rn)])

    @cached_property
    def inv(self):
        """
        Calculate the inverse matrix.
        The standard calculation by definition is quite slow and the time complexity is O(n^5).
        Here the inverse matrix is calculated using Guass-Jordan elimination.
        """
        N = self.order
        proto = self.to_list()
        AE = [proto[i] + i*[0] + [1] + (N-i-1)*[0] for i in range(N)]
        # Matrix upper-triangularization using linear transformation
        for k in range(N-1):
            x1 = AE[k]
            if x1[k] == 0:
                for xi in AE[k+1:]:
                    if xi[k] != 0:
                        xi[:], x1[:] = x1[:], xi[:]
                        break
                else: raise ZeroDivisionError("matrix is not invertable")

            for xi in AE[k+1:]:
                if xi[k] != 0:
                    R = xi[k]/x1[k]
                    xi[:] = [xij - R*x1j for xij, x1j in zip(xi, x1)]
        # Linearly transform the matrix from the opposite direction to diagonalize it
        for k in range(N-1, 0, -1):
            x1 = AE[k] # Row index is as the same
            if x1[k] == 0:
                # From negative column index one must deduct N (number of columns of the right half part)
                for xi in AE[k-1::-1]:
                    if xi[k] != 0:
                        xi[:], x1[:] = x1[:], xi[:]
                        break
                else: raise ZeroDivisionError("matrix is not invertable")

            for xi in AE[k-1::-1]:
                if xi[k] != 0:
                    R = xi[k]/x1[k]
                    xi[:] = [xij - R*x1j for xij, x1j in zip(xi, x1)]

        for k in range(N):
            AE_k = AE[k]
            AE_k[:] = [val/AE_k[k] for val in AE_k]

        return matrix([row[N:] for row in AE])

    def row_rank(self):
        N = self.rn
        proto = self.to_list()
        # Matrix upper-triangularization
        for k in range(N-1):
            x1 = proto[k]
            if x1[k] == 0:
                for xi in proto[k+1:]:
                    if xi[k] != 0:
                        xi[:], x1[:] = x1[:], xi[:]

            for xi in proto[k+1:]:
                if xi[k] != 0:
                    R = xi[k]/x1[k]
                    xi[:] = [xij - R*x1j for xij, x1j in zip(xi, x1)]
        return self.rn - self.__proto.count([0]*self.cn)

    def col_rank(self):
        return self.cn - self.t.row_rank()

    @cached_property
    def rank(self):
        return min([self.col_rank(), self.row_ranl()])

    @cached_property
    def rk(self): return self.rank

    @cached_property
    def t(self):
        """The transpose of a matrix"""
        return matrix(*zip(*self.__proto))

    @cached_property
    def T(self): return self.t

    @cached_property
    def h(self):
        """The conjugate transpose of the matrix (transpose and take conjugate complex number for each entry)."""
        return matrix(*zip(*chunked(map(lambda x: x.conjugate(), self.entry), self.cn)))

    def rot(self, time=1, clockwise=True):
        """Rotate the matrix by times of 90 degrees."""
        mtr = self
        if clockwise:
            for i in range(time%4):
                mtr = ~ mtr.t
        else:
            for i in range(time%4):
                mtr = reversed(mtr.t)
        return mtr

    @cached_property
    def density(self):
        """
        Density of matrix:
        The proportion of values not equal to 0.
        """
        return 1 - self.count(0) / len(self)


class matrixList(list):
    """
    This class is implemented mainly to show blocked matrices.
    The member within matrixList object must be matrices of the same row number.
    """
    def __init__(self, iterable_of_matrices):
        for item in iterable_of_matrices:
            assert isinstance(item, matrix)
        assert all_equal(iterable_of_matrices, lambda x: x.rn)
        self.__height = iterable_of_matrices[0].rn
        super().__init__(iterable_of_matrices)

    def __repr__(self):
        mtr_str = deque(zip(*map(lambda x: x.__str__().split("\n"), self)))
        if self.__height > 1:
            string = "\n┌" + "  ".join(mtr_str.popleft()) + "┐\n"
            for i in range(1, self.__height - 1):
                string += "│" + "  ".join(mtr_str.popleft()) + "│\n"
            else:
                string += "└" + ", ".join(mtr_str.popleft()) + "┘\n"
        else:
            string = "\n[" + ", ".join(mtr_str.popleft()) + "]\n"
        return string
        

def E(order=3):
    return matrix(*chunked(((1,) + (0,) * order) * (order - 1) + (1,), order))
