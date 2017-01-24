/*
 * Main function of Ground Station
 *
 * Author: Roice (LUO Bing)
 * Date: 
 *       2016-04-16 create this file (GSRAO)
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
#include "GSRAO_Config.h" // settings
#include "GSRAO_thread_comm.h"

/***************************************************************/
/**************************** MAIN *****************************/
/***************************************************************/

int main(int argc, char **argv) 
{
    /* initialize GS settings */
    GSRAO_Config_restore();
   
    /* initialize communication among threads */
    GSRAO_init_thread_comm();

    // Create a window for the display of the experiment data
    UI ui(700, 500, "Ground Station of Robot Active Olfaction System");
    
    // Run
    Fl::run();

    // save configs before closing
    GSRAO_Config_save();

    return 0;
}

#endif

/* End of main.cxx */
