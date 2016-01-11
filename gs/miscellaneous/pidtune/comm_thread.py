#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PID tuning tool for sniffer bee robot (MAV)
#                     Communication Thread Class
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

from threading import Thread

class CommunicationThread(Thread):

    wants_abort = False

    def run(self):
        while not self.wants_abort:
            # Send position & attitude info to Sniffer Bee
            # Message name: SBSP_FRESH_POS_MC
            #       (Sniffer Bee fresh position obtained from Motion Capture System)
            # Message ID: 0x3D
            sbsp_dir = '<'
            sbsp_size = chr(12)
            sbsp_cmd = chr(61)
            sbsp_data =


