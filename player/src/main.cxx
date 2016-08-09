/*
 * Main function of RAO Player
 *
 * Author: Roice (LUO Bing)
 * Date: 
 *       2016-08-07 create this file
 */

#include "FL/Fl.H"
#include "ui/UI.h" // control panel and GL view
#include "Player_Config.h" // settings

/***************************************************************/
/**************************** MAIN *****************************/
/***************************************************************/

int main(int argc, char **argv) 
{
    /* initialize settings */
    Config_restore();
    
    // Create a window for the display of the experiment data
    UI ui(700, 500, "Player of Robot Active Olfaction Experiments");
    
    // Run
    Fl::run();

    // save configs before closing
    Config_save();

    return 0;
}


/* End of main.cxx */
