/*
 * Main function of RAO Player
 *
 * Author: Roice (LUO Bing)
 * Date: 
 *       2016-08-07 create this file
 */

#include <config.h>
#if !HAVE_GL || !HAVE_GL_GLU_H
#include <FL/Fl.H>
#include <FL/fl_message.H>
int main(int, char**) {
  fl_alert("This demo does not work without GL and GLU");
  return 1;
}
#else
// end of added block

#include "FL/Fl.H"
#include "ui/UI.h" // control panel and GL view
#include "Config.h" // settings

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

#endif

/* End of main.cxx */
