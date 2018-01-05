"""
Utilities for Pandas
"""


from __future__ import absolute_import, division

from copy import deepcopy
import sys

import numpy as np
import pandas as pd
from pandas.api.types import (CategoricalDtype, is_categorical_dtype,
                              is_datetime64_any_dtype, is_dtype_equal,
                              is_object_dtype, is_timedelta64_dtype,
                              is_timedelta64_ns_dtype)


__all__ = ['INT_TYPES', 'UINT_TYPES', 'INT_DTYPES', 'UINT_DTYPES',
           'FLOAT_TYPES', 'FLOAT_DTYPES', 'BOOL_TYPES', 'BOOL_DTYPES',
           'convert_df_dtypes', 'convert_col_dtype', 'test_convert_col_dtype']


# NOTE: order must be ascending for both *_TYPES and *_DTYPES for some of the
# logic in the script to work! (see function `convert_df_dtypes`)
INT_TYPES = (np.int8, np.int16, np.int32, np.int64, int)
UINT_TYPES = (np.uint8, np.uint16, np.uint32, np.uint64)
FLOAT_TYPES = (np.float16, np.float32, np.float64, np.float128, float)
BOOL_TYPES = (np.bool, np.bool8, bool)

INT_DTYPES = tuple(sorted(set(np.dtype(t) for t in INT_TYPES)))
UINT_DTYPES = tuple(sorted(set(np.dtype(t) for t in UINT_TYPES)))
FLOAT_DTYPES = tuple(sorted(set(np.dtype(t) for t in FLOAT_TYPES)))
BOOL_DTYPES = tuple(sorted(set(np.dtype(t) for t in BOOL_TYPES)))


def wstderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def wstdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def convert_df_dtypes(df, **kwargs):
    """Convert dtype of each column of a DataFrame according to the logic in
    `convert_col_dtype`. Conversion is in-place, so the original DataFrame is
    overwritten. `kwargs` get passed along to that function, so see docs on
    that for more info about what is accepted for `kwargs`.

    Parameters
    ----------
    df : pandas.DataFrame

    """
    for col in df.columns:
        df[col] = convert_col_dtype(df[col], **kwargs)


def convert_col_dtype(col, int_to_category=True, force_fp32=True):
    """Convert datatypes for columns according to "sensible" rules for the
    tasks in this module:

    * integer types are reduced to smallest integer type without losing
      information, or to a categorical if that uses less memory (roughly)
    * float types are all made the same: either the type of the first element,
      or all are reduced to single precision
    * object types that contain strings are converted to categoricals
    * object types that contain numbers are converted according to the rules
      above to either floats, shortest-possible ints, or a categorical
    * bool types are forced to ``numpy.dtype('bool')``

    Parameters
    ----------
    col : pandas.Series
        Column

    int_to_category : bool
        Whether to convert integer types to categoricals in the case that this
        will save memory.

    force_fp32 : bool
        Force all floating-point data types to be single precision (fp32). If
        False, the type of the first element is used instead (for all values in
        the column).

    Returns
    -------
    col : pandas.Series

    """
    from pisa.utils.fileio import fsort

    categorical_dtype = CategoricalDtype()

    recognized_dtype = False
    original_dtype = col.dtype
    col_name = col.name

    if len(col) == 0: #pylint: disable=len-as-condition
        return col

    first_item = col.iloc[0]

    # Default: keep current dtype
    new_dtype = original_dtype

    if (is_categorical_dtype(original_dtype)
            or is_datetime64_any_dtype(original_dtype)
            or is_timedelta64_dtype(original_dtype)
            or is_timedelta64_ns_dtype(original_dtype)):
        recognized_dtype = True
        new_dtype = original_dtype
    elif is_object_dtype(original_dtype):
        if isinstance(first_item, basestring):
            recognized_dtype = True
            new_dtype = categorical_dtype
        # NOTE: Must check bool before int since bools look like ints (but not
        # vice versa)
        elif isinstance(first_item, BOOL_TYPES):
            recognized_dtype = True
            new_dtype = np.dtype('bool')
        elif isinstance(first_item, INT_TYPES + UINT_TYPES):
            recognized_dtype = True
            new_dtype = np.dtype('int')
        elif isinstance(first_item, FLOAT_TYPES):
            recognized_dtype = True
            new_dtype = np.dtype(type(first_item))

    # Convert ints to either shortest int possible or categorical,
    # whichever is smaller (use int if same size)
    if new_dtype in INT_DTYPES + UINT_DTYPES:
        recognized_dtype = True
        # See how large an int would be necessary
        col_min, col_max = col.min(), col.max()
        found_int_dtype = False
        int_dtype = None
        for int_dtype in INT_DTYPES:
            exponent = 8*int_dtype.itemsize - 1
            min_representable = -2 ** exponent
            max_representable = (2 ** exponent) - 1
            if col_min >= min_representable and col_max <= max_representable:
                found_int_dtype = True
                break
        if not found_int_dtype:
            raise ValueError('Value(s) in column "%s" exceed %s bounds'
                             % (col_name, int_dtype))

        # Check if categorical is probably smaller than int dtype; note that
        # the below is not perfect (i.e. is not based on exact internal
        # representation of categoricals in Pandas...) but should get us pretty
        # close, so that at least order-of-magnitude efficiencies will be
        # found)
        if int_to_category:
            num_unique = len(col.unique())
            category_bytes = int(np.ceil(np.log2(num_unique) / 8))
            if category_bytes < int_dtype.itemsize:
                new_dtype = categorical_dtype
            else:
                new_dtype = int_dtype

    elif new_dtype in FLOAT_DTYPES:
        recognized_dtype = True
        if force_fp32:
            new_dtype = np.dtype('float32')
        else:
            new_dtype = np.dtype(type(first_item))

    elif new_dtype in BOOL_DTYPES:
        recognized_dtype = True
        new_dtype = np.dtype('bool')

    if not recognized_dtype:
        wstderr('WARNING: Not modifying column "%s" with unhandled dtype "%s"'
                ' and/or sub-type "%s"\n'
                % (col_name, original_dtype.name, type(first_item)))

    if is_dtype_equal(new_dtype, original_dtype):
        if isinstance(first_item, basestring):
            return col.cat.reorder_categories(fsort(col.cat.categories))
        return col

    if is_categorical_dtype(new_dtype):
        new_col = col.astype('category')
        if isinstance(first_item, basestring):
            new_col.cat.reorder_categories(fsort(new_col.cat.categories),
                                           inplace=True)
        return new_col

    try:
        return col.astype(new_dtype)
    except ValueError:
        wstderr('WARNING: Could not convert column "%s" to dtype "%s"; keeping'
                ' original dtype "%s"\n'
                % (col_name, new_dtype, original_dtype))
        return col


def test_convert_col_dtype():
    """Test function `convert_col_dtype`. Also tests `convert_df_dtypes`."""

    categorical_dtype = CategoricalDtype()

    orig_df = pd.DataFrame([{
        '"int"': int(0),
        'np.int': np.int(1),
        'np.int16': np.int16(2),
        'np.int32': np.int32(2),
        'np.int64': np.int64(2),

        'np.uint': np.uint(1),
        'np.uint16': np.uint16(2),
        'np.uint32': np.uint32(2),
        'np.uint64': np.uint64(2),

        '"float"': float(0.0),
        'np.float': 0.1,
        'np.float32': np.float32(0.2),
        'np.float64': np.float64(0.3),

        '"bool"': False,
        'np.bool': np.bool(0),
        'np.bool8': np.bool8(0),

        '"object"#category': 'zero',
        '"object"#int': int(0),
        '"object"#float': float(0.0),
        '"object"#bool': False,

        '"category"': 'cat_zero',
    }])

    # TODO: test that the conversion of int vs. categorical works when an int
    # type is smaller than categorical
    # TODO: test with `datetime`, any other Pandas dtypes I missed...
    target_dtypes = {
        '"int"': (categorical_dtype, INT_TYPES),
        'np.int': (categorical_dtype, INT_TYPES),
        'np.int16': (categorical_dtype, INT_TYPES),
        'np.int32': (categorical_dtype, INT_TYPES),
        'np.int64': (categorical_dtype, INT_TYPES),

        'np.uint': (categorical_dtype, UINT_TYPES),
        'np.uint16': (categorical_dtype, UINT_TYPES),
        'np.uint32': (categorical_dtype, UINT_TYPES),
        'np.uint64': (categorical_dtype, UINT_TYPES),

        '"float"': (np.dtype('float32'), np.float32),
        'np.float': (np.dtype('float32'), np.float32),
        'np.float32': (np.dtype('float32'), np.float32),
        'np.float64': (np.dtype('float32'), np.float32),

        '"bool"': (np.dtype('bool'), BOOL_TYPES),
        'np.bool': (np.dtype('bool'), BOOL_TYPES),
        'np.bool8': (np.dtype('bool'), BOOL_TYPES),

        '"object"#category': (categorical_dtype, basestring),
        '"object"#int': (categorical_dtype, INT_TYPES),
        '"object"#float': (np.dtype('float32'), np.float32),
        '"object"#bool': (np.dtype('bool'), BOOL_TYPES),

        '"category"': (categorical_dtype, basestring)
    }

    # NOTE: for some reason, Pandas converts the data types even after I
    # explicitly set them with the `apply` method. Therefore, just manually
    # iterating column-by-column instead.

    #def _converter(col):
    #    try:
    #        new_col = col.astype(eval(col.name)) #pylint: disable=eval-used
    #    except:
    #        print('Col: "%s", first item: "%s"' % (col.name, col.iloc[0]))
    #        raise
    #    return new_col
    #orig_df = orig_df.apply(_converter, reduce=False, raw=True)

    for col in orig_df.columns:
        orig_df[col] = orig_df[col].astype(eval(col))

    new_df = deepcopy(orig_df)
    convert_df_dtypes(new_df)
    dtypes_dict = new_df.dtypes.to_dict()

    for col, dtype in dtypes_dict.items():
        target_dtype, target_subtype = target_dtypes[col]
        item = new_df[col].iloc[0]
        if dtype.name != target_dtype.name:
            raise TypeError(
                "Column '%s' has dtype '%s' but should be '%s'"
                % (col, dtype, target_dtype)
            )
        if not isinstance(item, target_subtype):
            raise TypeError(
                "Column '%s' has sub-type '%s' but should be (one of) '%s'"
                % (col, type(item), target_subtype)
            )

    return orig_df, new_df


if __name__ == '__main__':
    test_convert_col_dtype()
