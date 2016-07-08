/*
 * Robot Active Olfation Method Gallery
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.06.02
 */

#include "method/method.h"
#include "method/hover_measure.h"
#include "method/back_forth_measure.h"

static methodName_e current_method = METHOD_NONE;

bool method_start(methodName_e method_name)
{
    bool result = false;

    switch (method_name)
    {
        case METHOD_HOVER_MEASURE:
            result = hover_measure_init();
            if (result)
                current_method = METHOD_HOVER_MEASURE;
            break;
        case METHOD_BACK_FORTH_MEASURE:
            result = back_forth_measure_init();
            if (result)
                current_method = METHOD_BACK_FORTH_MEASURE;
            break;
        default:
            break;
    }

    return result;
}

void method_stop(void)
{
    switch (current_method)
    {
        case METHOD_HOVER_MEASURE:
            hover_measure_stop();
            current_method = METHOD_NONE;
            break;
        case METHOD_BACK_FORTH_MEASURE:
            back_forth_measure_stop();
            current_method = METHOD_NONE;
            break;
        default:
            break;
    }
}
