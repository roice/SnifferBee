#ifndef FOC_WIND_H
#define FOC_WIND_H

void foc_wind_smooth_init(std::vector<FOC_Wind_t>& out);
void foc_wind_smooth_update(FOC_Input_t& new_in, std::vector<FOC_Wind_t>& out);

#endif
