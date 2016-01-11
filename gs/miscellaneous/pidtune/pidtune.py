#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from traits.api import HasTraits, Instance, Button, String, Int
from traitsui.api import View, Item, Group, VSplit, ButtonEditor, TextEditor,\
        RangeEditor, Handler

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
    # indicate whether the connection is on or not
    text_connect_or_not = String
    # Altitude PID value
    pid_alt_p = Int
    pid_alt_i = Int
    pid_alt_d = Int
    # Position PID value
    pid_pos_p = Int
    pid_pos_i = Int
    pid_pos_d = Int
    # Position Rate PID value
    pid_posr_p = Int
    pid_posr_i = Int
    pid_posr_d = Int
    # send pid value to MAV
    button_send_pid_value = Button("Send PID value to MAV")

    # =========== Other ===========
    # Communication thread
    comm_thread = Instance(CommunicationThread)

    ###################################
    # view
    view = View(
            VSplit(
                Group(
                    # MAV and Mocap communication related
                    Item('button_connect', show_label = False),
                    Item('text_connect_or_not',
                        editor = TextEditor(auto_set = False,
                                            enter_set = False),
                        show_label = False, style = 'readonly'),
                    show_border = True),
                Group(
                    # PID tuning
                    Item('pid_alt_p',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Altitude P'),
                    Item('pid_alt_i',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Altitude I'),
                    Item('pid_alt_d',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Altitude D'),
                    Item('pid_pos_p',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Position P'),
                    Item('pid_pos_i',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Position I'),
                    Item('pid_pos_d',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Position D'),
                    Item('pid_posr_p',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Pos Rate P'),
                    Item('pid_posr_i',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Pos Rate I'),
                    Item('pid_posr_d',
                        editor = RangeEditor(   low = '0',
                                                high = '255',
                                                format = '%d',
                                                mode = 'slider'),
                        label = 'Pos Rate D'),
                    Item('button_send_pid_value', show_label=False),
                    show_border = True,
                    ),
                )
            )

    def _text_connect_or_not_default(self):
        return '******** Not Connected ********'

    def _pid_alt_p_default(self):
        return 50
    def _pid_alt_i_default(self):
        return 0
    def _pid_alt_d_default(self):
        return 0
    def _pid_pos_p_default(self):
        return 15
    def _pid_pos_i_default(self):
        return 0
    def _pid_pos_d_default(self):
        return 0
    def _pid_posr_p_default(self):
        return 34
    def _pid_posr_i_default(self):
        return 14
    def _pid_posr_d_default(self):
        return 53

    def _button_connect_fired(self):
        if self.comm_thread and self.comm_thread.isAlive():
            # kill communication thread if it's running
            self.comm_thread.wants_abort = True
        else:
            # start communication thread
            self.comm_thread.start()


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
            title = 'PID Tuning for Sniffer Robots',
            height = 0.5, width = 0.1,
            handler = MainWindowHandler(),
            )

###############################################################################
# Execute if running this script
if __name__ == '__main__':
    app = MainWindow()
    app.configure_traits()

