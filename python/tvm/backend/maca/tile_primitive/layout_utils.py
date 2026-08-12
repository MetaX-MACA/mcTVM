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
"""ComposeLayout helpers shared by MACA tile-primitive dispatches."""

import math

from tvm.tirx.layout import ComposeLayout, S, TileLayout


def recompose_swizzle(original_layout, transformed_tile: TileLayout):
    """Replace a ComposeLayout's tile while preserving its swizzle parameters."""
    if isinstance(original_layout, ComposeLayout):
        return ComposeLayout(
            int(original_layout.per_element),
            int(original_layout.swizzle_len),
            int(original_layout.atom_len),
            transformed_tile,
            bool(original_layout.swizzle_inner),
        )
    return transformed_tile


def strip_swizzle_to_tile(layout, get_extents):
    """Return the tile portion of a layout for grouping and slicing.

    A bare swizzle is now represented as a ComposeLayout over a trivial tile.
    Its tile may describe only the swizzle period, so rebuild an identity tile
    over the transfer extents when the two sizes differ.
    """
    if not isinstance(layout, ComposeLayout):
        return layout

    tile = layout.tile_layout
    if not tile.is_trivial():
        return tile

    try:
        extents = get_extents()
        tile_size = int(tile.size())
        layout_size = math.prod(int(extent) for extent in extents)
    except (TypeError, ValueError):
        return tile

    if tile_size != layout_size:
        return TileLayout(S[tuple(extents)])
    return tile
