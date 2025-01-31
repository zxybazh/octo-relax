# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return, invalid-name, unused-import
"""The expression nodes of Relay."""
from __future__ import absolute_import

import tvm._ffi
from tvm.runtime import convert
from tvm.ir import BaseFunc

from .expr import Call
from . import _ffi_api


@tvm._ffi.register_object("relay.Function")
class Function(BaseFunc):
    """A function declaration expression.

    Parameters
    ----------
    params: List[tvm.relay.Var]
        List of input parameters to the function.

    body: tvm.relay.Expr
        The body of the function.

    ret_type: Optional[tvm.relay.Type]
        The return type annotation of the function.

    type_params: Optional[List[tvm.relay.TypeParam]]
        The additional type parameters, this is only
        used in advanced usecase of template functions.

    span: Optional[tvm.relay.Span]
        Span that points to original source code.
    """

    def __init__(self, params, body, ret_type=None, type_params=None, attrs=None, span=None):
        if type_params is None:
            type_params = convert([])

        self.__init_handle_by_constructor__(
            _ffi_api.Function, params, body, ret_type, type_params, attrs, span
        )

    def __call__(self, *args):
        """Invoke the global function.

        Parameters
        ----------
        args: List[relay.Expr]
            Arguments.
        """
        return Call(self, args, None, None)


@tvm._ffi.register_func("relay.FunctionWithFields")
def FunctionWithFields(
    function,
    params=None,
    body=None,
    ret_type=None,
    ty_params=None,
    attrs=None,
    virtual_device=None,
    span=None,
):
    """
    Returns function with the given properties. A None property denotes 'no change'.
    Returns function if all properties are unchanged. Otherwise, returns a copy with the new
    fields.
    """
    return _ffi_api.FunctionWithFields(
        function, params, body, ret_type, ty_params, attrs, virtual_device, span
    )
