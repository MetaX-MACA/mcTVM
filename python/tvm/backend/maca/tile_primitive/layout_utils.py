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

import functools
import math
import operator

from tvm.arith import Analyzer
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


def get_sublayout_from_region(layout, buffer_shape, region_st, region_extent):
    """Return the layout sliced to a buffer region when slicing succeeds."""
    if not layout:
        return layout
    region = [(region_st[i], region_st[i] + region_extent[i]) for i in range(len(region_st))]
    sliced = layout.slice(list(buffer_shape), region)
    return sliced if sliced is not None else layout


def get_local_region(orig_layout: TileLayout, buffer_shape, region_st, region_extent):
    """Return storage-local shape, start, and extent for a contiguous region.

    Thread-partitioned dimensions must be selected in full.  This gives local
    tile-primitive emitters a physical-local view that matches a sliced layout.
    """
    grouped, seps = orig_layout.group(list(buffer_shape))
    local_shape = []
    local_st = []
    local_ext = []
    analyzer = Analyzer()

    for dim in range(len(buffer_shape)):
        shard_range = list(range(seps[dim], seps[dim + 1]))
        has_local = any(not grouped.shard[index].axis.is_thread() for index in shard_range)
        if not has_local:
            continue

        has_thread = any(grouped.shard[index].axis.is_thread() for index in shard_range)
        if not has_thread:
            local_shape.append(buffer_shape[dim])
            local_st.append(region_st[dim])
            local_ext.append(region_extent[dim])
            continue

        def decompose(value):
            coords = []
            remaining = value
            for position, _ in enumerate(shard_range):
                suffix_product = functools.reduce(
                    operator.mul,
                    [
                        grouped.shard[shard_range[inner]].extent
                        for inner in range(position + 1, len(shard_range))
                    ],
                    1,
                )
                coords.append(remaining // suffix_product)
                remaining = remaining % suffix_product
            return coords

        start_coords = decompose(region_st[dim])
        end_coords = decompose(region_st[dim] + region_extent[dim] - 1)
        current_shape = 1
        current_start = 0
        current_end = 0
        for position in reversed(range(len(start_coords))):
            iterator = grouped.shard[seps[dim] + position]
            if iterator.axis.is_thread():
                if not (
                    analyzer.can_prove_equal(start_coords[position], 0)
                    and analyzer.can_prove_equal(end_coords[position], iterator.extent - 1)
                ):
                    return None
                continue
            spans_two_coords = analyzer.can_prove_equal(
                end_coords[position] - start_coords[position], 1
            )
            selects_full_extent = analyzer.can_prove_equal(
                start_coords[position], 0
            ) and analyzer.can_prove_equal(end_coords[position], iterator.extent - 1)
            if not spans_two_coords and not selects_full_extent:
                return None
            current_shape *= iterator.extent
            current_start = current_start * iterator.extent + start_coords[position]
            current_end = current_end * iterator.extent + end_coords[position]

        assert analyzer.can_prove_equal(
            region_extent[dim],
            functools.reduce(
                operator.mul,
                [end - start + 1 for start, end in zip(start_coords, end_coords)],
                1,
            ),
        )
        local_shape.append(current_shape)
        local_st.append(current_start)
        local_ext.append(current_end - current_start + 1)

    if not local_shape:
        return [1], [0], [1]
    return local_shape, local_st, local_ext
