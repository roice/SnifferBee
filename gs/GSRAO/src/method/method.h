/*
 * Robot Active Olfation Method Gallery
 *
 * Author:
 *      Roice Luo (Bing Luo)
 * Date:
 *      2016.06.02
 */

typedef enum {
    METHOD_HOVER_MEASURE = 0,
    METHOD_BACK_FORTH_MEASURE = 1,
    METHOD_CIRCLE_MEASURE = 2,
    METHOD_FLYING_COMPASS = 3,
    METHOD_ODOR_COMPASS = 4,
    METHOD_ITEM_COUNT,
    METHOD_NONE
} methodName_e;

bool method_start(methodName_e);

void method_stop(void);
