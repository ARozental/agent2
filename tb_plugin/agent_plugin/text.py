from tensorboard import plugin_util
import numpy as np
import textwrap

WARNING_TEMPLATE = textwrap.dedent(
    """\
  **Warning:** This text summary contained data of dimensionality %d, but only \
  2d tables are supported. Showing a 2d slice of the data instead."""
)


def reduce_to_2d(arr):
    """Given a np.npdarray with nDims > 2, reduce it to 2d.
    It does this by selecting the zeroth coordinate for every dimension greater
    than two.
    Args:
      arr: a numpy ndarray of dimension at least 2.
    Returns:
      A two-dimensional subarray from the input array.
    Raises:
      ValueError: If the argument is not a numpy ndarray, or the dimensionality
        is too low.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("reduce_to_2d requires a numpy.ndarray")

    ndims = len(arr.shape)
    if ndims < 2:
        raise ValueError("reduce_to_2d requires an array of dimensionality >=2")
    # slice(None) is equivalent to `:`, so we take arr[0,0,...0,:,:]
    slices = ([0] * (ndims - 2)) + [slice(None), slice(None)]
    return arr[slices]


def text_array_to_html(text_arr, enable_markdown):
    """Take a numpy.ndarray containing strings, and convert it into html.
    If the ndarray contains a single scalar string, that string is converted to
    html via our sanitized markdown parser. If it contains an array of strings,
    the strings are individually converted to html and then composed into a table
    using make_table. If the array contains dimensionality greater than 2,
    all but two of the dimensions are removed, and a warning message is prefixed
    to the table.
    Args:
      text_arr: A numpy.ndarray containing strings.
      enable_markdown: boolean, whether to enable Markdown
    Returns:
      The array converted to html.
    """
    if not text_arr.shape:
        # It is a scalar. No need to put it in a table.
        return text_arr.item()
    warning = ""
    if len(text_arr.shape) > 2:
        warning = plugin_util.markdown_to_safe_html(
            WARNING_TEMPLATE % len(text_arr.shape)
        )
        text_arr = reduce_to_2d(text_arr)

    # Convert utf-8 bytes to str. The built-in np.char.decode doesn't work on
    # object arrays, and converting to an numpy chararray is lossy.
    decode = lambda bs: bs.decode("utf-8") if isinstance(bs, bytes) else bs
    text_arr_str = np.array(
        [decode(bs) for bs in text_arr.reshape(-1)]
    ).reshape(text_arr.shape)
    table = [text_arr_str[i] for i in range(text_arr_str.shape[0])]
    return warning + ''.join(table)


def process_event(wall_time, step, string_ndarray, enable_markdown):
    """Convert a text event into a JSON-compatible response."""
    html = text_array_to_html(string_ndarray, enable_markdown)
    return {
        "wall_time": wall_time,
        "step": step,
        "text": html,
    }
