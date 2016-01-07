#!/usr/bin/python
# coding=utf-8
#
# PID tuning tool for sniffer bee robot (MAV)
#
# Author: Roice Luo <oroice@foxmail.com>
# copyright (c) 2015 Roice Luo <https://github.com/roice>
#
# This routine is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2.1 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

"""
    Documentation is not available yet, please browse the source code by yourself~
"""

# Enthought imports
from traits.api import HasTraits, Instance, Button
from traitsui.api import View, Item, Group, VSplit, ButtonEditor, Handler

###############################################################################
# GUI elements
#-------------

# Control panel
class Panel(HasTraits):
    '''
    This object is the core of the PID tuning tool interface.
    '''

    # =========== Panel GUI ===========
    # establish/break connection to MAVs and Motion Capture System
    button_connect = Button("Connect to MAVs & Mocap")

    ###################################
    # view
    view = View(
                # MAV and Mocap communication related
                Group(
                    Item('button_connect', show_label = False),
                    label = 'Communication to MAVs & Mocap', show_border=True,),
            )

# Handler for Main window
class MainWindowHandler(Handler):
    def close(self, info, is_OK):
        print 'PID tuning tool closed'
        return True

# Main window
class MainWindow(HasTraits):
    panel = Instance(Panel)

    def _panel_default(self):
        return Panel()

    view = View(
            Item('panel', style = 'custom', show_label = False),
            resizable = True,
            title = 'PID Tuning Tool for Sniffer Robots',
            height = 0.3, width = 0.3,
            handler = MainWindowHandler(),
            )

###############################################################################
# Execute if running this script
if __name__ == '__main__':
    app = MainWindow()
    app.configure_traits()

