#ifndef NET_SEND_BEAR_CMD_H
#define NET_SEND_BEAR_CMD_H

bool net_send_bear_cmd_init(const char* local_address);
void net_send_bear_cmd_goto(float, float, float);
void net_send_bear_cmd_terminate(void);

#endif
