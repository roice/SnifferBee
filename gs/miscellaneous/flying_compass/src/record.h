/*
 * Record data
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.06.21
 */

#ifndef RECORD_H
#define RECORD_H

#include <vector>
#include "foc/flying_odor_compass.h"

void Record_Data(std::vector<FOC_Wind_t>&, std::vector<FOC_Input_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_Reading_t>&, std::vector<FOC_Delta_t>&,
        std::vector<FOC_Estimation_t>&);

#endif
