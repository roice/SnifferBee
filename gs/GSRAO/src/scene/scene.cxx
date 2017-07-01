/* This file defines the outline of the experiment area
 * When the robot reaches the outline, it will be forced to stay there
 * 
 * Author:      Roice Luo
 * Date:        2017.06.23
 */

#include <cmath>
#include <string.h>
#include "robot/robot.h"

/* horizontal experimental area
 * defined by a polygon
 */
static int polySides = 5;
static float polyX[5] = {3.0, -1.2, -2.6, -2.6, 3.0};
static float polyY[5] = {2.0, 2.0, 0.0, -3.0, -3.0};
/* vertical space of the experiment area */
static float height_range[2] = {0.3, 2.5};

/* gas source position
 * for the use of shutting down robots when they reached source
 */
//static float gas_source_pos[3] = {3.2, 0.0, 1.35}; // source at door
static float gas_source_pos[3] = {-0.2, -3.4, 1.35}; // source at window

// Globals which should be set before calling this function:
//
// int    polySides  =  how many corners the polygon has
// float  polyX[]    =  horizontal coordinates of corners
// float  polyY[]    =  vertical coordinates of corners
// float  x,y        =  point to be tested
//
//  Thefunction will return TRUE if the point x,y is inside the polygon, or
//  FALSE if it is not.  If the point is exactly on the edge of the polygon,
// then the function may return TRUE or FALSE.
//
// Note that division by zero is avoided because the division is protected
//  by the "if" clause which surrounds it.

bool pointInPolygon (float x, float y)
{
    int i, j = polySides-1;
    bool  oddNodes = false;

    for (i = 0; i < polySides; i++) {
        if( (polyY[i] < y && polyY[j] >= y || polyY[j] < y && polyY[i] >= y)
            && (polyX[i] <= x || polyX[j] <= x)) {
            oddNodes ^= (polyX[i]+(y-polyY[i])/(polyY[j]-polyY[i])*(polyX[j]-polyX[i])<x);
        }
        j=i;
    }
    return oddNodes;
}

bool scene_pos_is_inside_exp_area(float *pos)
{
    if (!pointInPolygon(pos[0], pos[1]))
        return false;
    if (pos[2] > height_range[0] and pos[2] < height_range[1])
        return true;
    else
        return false;
}

bool scene_change_ref_pos(int robot_idx, float *pos)
{
    Robot_Ref_State_t* robot_ref = robot_get_ref_state(); // get robot ref state

    /* check if the desired ref pos will reach the source */
    if (std::sqrt(std::pow(pos[0]-gas_source_pos[0],2)+std::pow(pos[1]-gas_source_pos[1],2)) < 1.0) {
        printf("Reached source, shutting down robot.\n");
        robot_shutdown(); // the ref pos has reached the source, experiment done
        return true;
    }

    /* check if the desired ref pos is outside the experiment area */
    if (scene_pos_is_inside_exp_area(pos)) {
        memcpy(robot_ref[robot_idx].enu, pos, 3*sizeof(float));
        return true;
    }
    else
    {// stay here
        return false;
    }
}
