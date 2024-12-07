#!BPY

"""
Name: 'Wedge'
Blender: 245
Group: 'AddMesh'
Tip: 'Add Wedge Object...'
"""
import Blender
import BPyAddMesh
__author__ = ["Four Mad Men", "FourMadMen.com"]
__version__ = '1.00'
__url__ = ["Script, http://www.fourmadmen.com/blender/scripts/AddMesh/wedge/add_mesh_wedge.py",
           "Script Index, http://www.fourmadmen.com/blender/scripts/index.html", "Author Site , http://www.fourmadmen.com"]
__email__ = ["bwiki {at} fourmadmen {dot} com"]


__bpydoc__ = """

Usage:

* Launch from Add Mesh menu

* Modify parameters as desired or keep defaults

"""

# ***** BEGIN GPL LICENSE BLOCK *****
#
# Copyright (C) 2008, FourMadMen.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#
# ***** END GPL LICENCE BLOCK *****


def add_wedge(PREF_WIDTH, PREF_HEIGHT, PREF_DEPTH):
    Vector = Blender.Mathutils.Vector

    verts = []
    faces = []

    half_width = PREF_WIDTH * .5
    half_height = PREF_HEIGHT * .5
    half_depth = PREF_DEPTH * .5

    verts.append(Vector(-half_width, -half_height, half_depth))
    verts.append(Vector(-half_width, -half_height, -half_depth))

    verts.append(Vector(half_width, -half_height, half_depth))
    verts.append(Vector(half_width, -half_height, -half_depth))

    verts.append(Vector(-half_width, half_height, half_depth))
    verts.append(Vector(-half_width, half_height, -half_depth))

    faces.append((0, 2, 4))
    faces.append((1, 3, 5))
    faces.append((0, 1, 3, 2))
    faces.append((0, 4, 5, 1))
    faces.append((2, 3, 5, 4))

    return verts, faces


def main():
    Draw = Blender.Draw
    PREF_WIDTH = Draw.Create(2.0)
    PREF_HEIGHT = Draw.Create(2.0)
    PREF_DEPTH = Draw.Create(2.0)

    if not Draw.PupBlock('Add Wedge', [
        ('Width:', PREF_WIDTH,  0.01, 100, 'Width of Wedge'),
        ('Height:', PREF_HEIGHT,  0.01, 100, 'Height of Wedge'),
        ('Depth:', PREF_DEPTH,  0.01, 100, 'Depth of Wedge'),
    ]):
        return

    verts, faces = add_wedge(PREF_WIDTH.val, PREF_HEIGHT.val, PREF_DEPTH.val)

    BPyAddMesh.add_mesh_simple('Wedge', verts, [], faces)


main()
