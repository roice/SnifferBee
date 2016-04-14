This project aims to produce a module to generate PPM singnals for TX modules. The module receives commands from UART and produce PPM signals, one PPM signal for a TX module.

   _________________
  |                 |
  |                 |
  |                 |                               _____________
  |             GPIO|_____PPM_signal_for_UAV_1_____| TX module 1
  |                 |                               _____________
  |             GPIO|_____PPM_signal_for_UAV_2_____| TX module 2
  |                 |                               _____________
  |             GPIO|_____PPM_signal_for_UAV_3_____| TX module 3
  |                 |                               _____________
  |             GPIO|_____PPM_signal_for_UAV_4_____| TX module 4
  |                 |       ___________
  |                 |      |           |
  |         USART RX|______|TX of UART |
  |                 |      |           |
  |                 |      |           |
  |_STM32_Board_____|      |_PC________|
