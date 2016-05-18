/*
 * User Interface of Ground Station
 *         using FLTK
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-04-16 create this file
 */

/* FLTK */
#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Light_Button.H>
#include <FL/Fl_Pixmap.H>
#include <FL/Fl_Tabs.H>
#include <FL/fl_ask.H>
#include <FL/Fl_Shared_Image.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Input.H>
/* OpenGL */
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <FL/glut.H>
/* GSRAO */
#include "ui/UI.h"
#include "ui/icons/icons.h" // pixmap icons used in Tool bar
#include "ui/View.h" // 3D RAO view
#include "ui/widgets/Fl_LED_Button/Fl_LED_Button.H"
#include "GSRAO_Config.h"
/* Linux Network */
#include <ifaddrs.h>
#include <sys/socket.h>
#include <arpa/inet.h>

/*------- Configuration Dialog -------*/
struct ConfigDlg_Widgets { // for parameter saving
    // number of robots
    Fl_Choice* scenario_num_of_robots;
    // network interface for receiving multicast info from Motive software (Opti-Track)
    Fl_Choice* mocap_netcard;
    // rigid body number of robots
    Fl_Choice* mocap_rigid_body_num_of_robot[4]; // 4 robots max
    // serial port transmitting PPM signals (frames)
    Fl_Input* ppmcnt_serial_port;
    // serial port receiving data
    Fl_Input* dnet_serial_port;
};
class ConfigDlg : public Fl_Window
{
public:
    ConfigDlg(int xpos, int ypos, int width, int height, const char* title); 
    // widgets
    struct ConfigDlg_Widgets ws;
private:
    // callback funcs
    static void cb_close(Fl_Widget*, void*);
    static void cb_switch_tabs(Fl_Widget*, void*);
    static void cb_change_num_of_robots(Fl_Widget*, void*);
    // function to save current value of widgets to runtime configs
    static void save_value_to_configs(ConfigDlg_Widgets*);
    // function to get runtime configs to set value of widgets
    static void get_value_from_configs(ConfigDlg_Widgets*);
};

void ConfigDlg::cb_close(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        // save widget values to GSRAO runtime configs when closing the dialog window
        struct ConfigDlg_Widgets *ws = (struct ConfigDlg_Widgets*)data;
        save_value_to_configs(ws);
        // close dialog
        ((Fl_Window*)w)->hide();
    }
}

void ConfigDlg::cb_switch_tabs(Fl_Widget *w, void *data)
{
    Fl_Tabs *tabs = (Fl_Tabs*)w; 
    // When tab changed, make sure it has same color as its group
    tabs->selection_color( (tabs->value())->color() );
}

void ConfigDlg::cb_change_num_of_robots(Fl_Widget* w, void* data)
{
    struct ConfigDlg_Widgets *ws = (struct ConfigDlg_Widgets*)data;

    // deactivate & activate corresponding mocap rigid body selections
    for (char i = ws->scenario_num_of_robots->value()+1; i < 4; i++) // 4 robots max
        ws->mocap_rigid_body_num_of_robot[i]->deactivate();
    for (char i = 0; i <= ws->scenario_num_of_robots->value(); i++)
        ws->mocap_rigid_body_num_of_robot[i]->activate();
}

void ConfigDlg::save_value_to_configs(ConfigDlg_Widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
    // Robot
    configs->robot.num_of_robots = ws->scenario_num_of_robots->value()+1; // Fl_Choice count from 0
    configs->robot.ppm_serial_port_path = ws->ppmcnt_serial_port->value(); // serial port path for PPM
    configs->robot.dnet_serial_port_path = ws->dnet_serial_port->value(); // serial port path for data
    // Link
    configs->mocap.netcard = ws->mocap_netcard->value(); // save netcard num
    for (char i = 0; i < 4; i++) // 4 robots max
        configs->mocap.rigid_body_num_of_robot[i] = ws->mocap_rigid_body_num_of_robot[i]->value(); // save rigid body index
}

void ConfigDlg::get_value_from_configs(ConfigDlg_Widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
}

ConfigDlg::ConfigDlg(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs

    // add event handle to dialog window
    callback(cb_close, (void*)&ws);   
    // begin adding children
    begin();
    // Tabs
    int t_x = 5, t_y = 5, t_w = w()-10, t_h = h()-10;
    Fl_Tabs *tabs = new Fl_Tabs(t_x, t_y, t_w, t_h);
    {
        tabs->callback(cb_switch_tabs); // callback func when switch tabs

        // Tab Scenario
        Fl_Group *scenario = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Scenario");
        {
            // color of this tab
            scenario->color(0xebf4fa00); // water
            scenario->selection_color(0xebf4fa00); // water

            // number of robots
            ws.scenario_num_of_robots = new Fl_Choice(t_x+10+160, t_y+25+10, 100, 25,"Number of robots ");
            ws.scenario_num_of_robots->add("1");
            ws.scenario_num_of_robots->add("2");
            ws.scenario_num_of_robots->add("3");
            ws.scenario_num_of_robots->add("4");
            ws.scenario_num_of_robots->value(configs->robot.num_of_robots-1); // Fl_Choice count from 0
            ws.scenario_num_of_robots->callback(cb_change_num_of_robots, (void*)&ws);
        }
        scenario->end();

        // Tab Link
        Fl_Group *link = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Link");
        {
            // color of this tab
            link->color(0xe8e8e800); // light milk tea
            link->selection_color(0xe8e8e800); // light milk tea

            // Control PPM signals
            Fl_Box *ppmcnt = new Fl_Box(t_x+10, t_y+25+10, 370, 65,"PPM Control Signals");
            ppmcnt->box(FL_PLASTIC_UP_FRAME);
            ppmcnt->labelsize(16);
            ppmcnt->labelfont(FL_COURIER_BOLD_ITALIC);
            ppmcnt->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            //   Set serial port transmitting the ppm control signals(frames)
            ws.ppmcnt_serial_port = new Fl_Input(t_x+10+100, t_y+25+10+30, 200, 25, "Serial Port ");
            ws.ppmcnt_serial_port->value(configs->robot.ppm_serial_port_path.c_str());

            // Data network
            Fl_Box *dnet = new Fl_Box(t_x+10, t_y+25+10+70, 370, 65,"Data Network");
            dnet->box(FL_PLASTIC_UP_FRAME);
            dnet->labelsize(16);
            dnet->labelfont(FL_COURIER_BOLD_ITALIC);
            dnet->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            //   Set serial port receiving the data
            ws.dnet_serial_port = new Fl_Input(t_x+10+100, t_y+25+10+100, 200, 25, "Serial Port ");
            ws.dnet_serial_port->value(configs->robot.dnet_serial_port_path.c_str());

            // Motion capture settings
            Fl_Box *mocap = new Fl_Box(t_x+10, t_y+25+10+140, 370, 130,"Motion Capture");
            mocap->box(FL_PLASTIC_UP_FRAME);
            mocap->labelsize(16);
            mocap->labelfont(FL_COURIER_BOLD_ITALIC);
            mocap->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            //   Select network interface receiving the multicast info of Motion Capture System
            ws.mocap_netcard = new Fl_Choice(t_x+10+70, t_y+25+10+30+140, 280, 25, "Netcard");
            //     get all network interfaces for choosing
            struct ifaddrs* ifAddrStruct = NULL;
            struct ifaddrs* ifa = NULL;
            void* tmpAddrPtr = NULL;
            char ncName[100];
            char netcard_count = 0;
            getifaddrs(&ifAddrStruct);
            for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
                if (!ifa->ifa_addr) {
                continue;
                }
                if (ifa->ifa_addr->sa_family == AF_INET) { // check it is IP4
                    // is a valid IP4 Address
                    tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
                    char addressBuffer[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
                    // add this net interface to choice list
                    snprintf(ncName, 100, "%s %s IPv4", ifa->ifa_name, addressBuffer);
                    ws.mocap_netcard->add(ncName);
                    // count netcard number
                    netcard_count++;
                } else if (ifa->ifa_addr->sa_family == AF_INET6) { // check it is IP6
                    // is a valid IP6 Address
                    tmpAddrPtr=&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
                    char addressBuffer[INET6_ADDRSTRLEN];
                    inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
                    // add this net interface to choice list
                    snprintf(ncName, 100, "%s %s IPv6", ifa->ifa_name, addressBuffer);
                    ws.mocap_netcard->add(ncName);
                    // count netcard number
                    netcard_count++;
                } 
            }
            if (ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);
            //    default choice
            if (configs->mocap.netcard < netcard_count)
                ws.mocap_netcard->value(configs->mocap.netcard);
            else
                ws.mocap_netcard->value(0);
            ws.mocap_netcard->tooltip("Select which network interface receives multicast info from Motive software");

            //   Config rigid body index for microdrones
            const char* robot_name[] = {"robot 1", "robot 2", "robot 3", "robot 4"};
            char rbName[20];
            for (char i = 0; i < 4; i++) // 4 robots max
            {
                ws.mocap_rigid_body_num_of_robot[i] = new Fl_Choice(t_x+10+70+175*(i%2), t_y+25+10+60+30*(i<2?0:1)+140, 120, 25, robot_name[i]);
                for (char j = 0; j < 10; j++) // 10 rigid body candidates
                {
                    snprintf(rbName, 20, "rigid body %d", j+1);
                    ws.mocap_rigid_body_num_of_robot[i]->add(rbName);
                    ws.mocap_rigid_body_num_of_robot[i]->tooltip("Select corresponding rigid body number of the robot");
                }
                // set choice according to configs
                ws.mocap_rigid_body_num_of_robot[i]->value(configs->mocap.rigid_body_num_of_robot[i]);
                // activate/deactivate according to number of robots
                if (i <= ws.scenario_num_of_robots->value())
                    ws.mocap_rigid_body_num_of_robot[i]->activate();
                else
                    ws.mocap_rigid_body_num_of_robot[i]->deactivate();
            }
        }
        link->end();

        // Tab Flow
        Fl_Group *flow = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Flow");
        {
            // color of this tab
            flow->color(0xe0ffff00); // light blue
            flow->selection_color(0xe0ffff00); // light blue

            // Mean wind velocity
            Fl_Box *m_wind = new Fl_Box(t_x+10, t_y+25+10, 370, 130,"Mean Wind Vel");
            m_wind->box(FL_PLASTIC_UP_FRAME);
            m_wind->labelsize(16);
            m_wind->labelfont(FL_COURIER_BOLD_ITALIC);
            m_wind->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            // Mean wind velocity x/y/z components
            Fl_Value_Slider *m_wind_x = new Fl_Value_Slider(t_x+10+30,t_y+25+10+30,300,25,"X");
            Fl_Value_Slider *m_wind_y = new Fl_Value_Slider(t_x+10+30,t_y+25+10+60,300,25,"Y");
            Fl_Value_Slider *m_wind_z = new Fl_Value_Slider(t_x+10+30,t_y+25+10+90,300,25,"Z");
            m_wind_x->labelsize(16);
            m_wind_y->labelsize(16);
            m_wind_z->labelsize(16);
            m_wind_x->type(FL_HOR_NICE_SLIDER);
            m_wind_y->type(FL_HOR_NICE_SLIDER);
            m_wind_z->type(FL_HOR_NICE_SLIDER);
            m_wind_x->align(Fl_Align(FL_ALIGN_LEFT));
            m_wind_y->align(Fl_Align(FL_ALIGN_LEFT));
            m_wind_z->align(Fl_Align(FL_ALIGN_LEFT));
            m_wind_x->bounds(0, 10); // 0~10 m/s
            m_wind_y->bounds(0, 10);
            m_wind_z->bounds(0, 10);
            new Fl_Box(t_x+10+30+300,t_y+25+10+30, 30, 25, "m/s");
            new Fl_Box(t_x+10+30+300,t_y+25+10+60, 30, 25, "m/s");
            new Fl_Box(t_x+10+30+300,t_y+25+10+90, 30, 25, "m/s");
        }
        flow->end();
        // Tab Plume
        Fl_Group *plume = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Plume");
        {
            // color of this tab
            plume->color(0xeeee0000); // light yellow+green (chlorine)
            plume->selection_color(0xeeee0000); // light yellow+green
        }
        plume->end();
        // Tab Robot
        Fl_Group *robot = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Robot");
        {
            // color of this tab
            robot->color(0xa8a8a800); // light yellow+green (chlorine)
            robot->selection_color(0xa8a8a800); // light yellow+green
        }
        robot->end();
    }
    // Make sure default tab has same color as its group
    tabs->selection_color( (tabs->value())->color() );
    tabs->end();
    
    end();
    // set widget value according to runtime configs
    get_value_from_configs(&ws);
    show();
}

/*------- Remote controller panel -------*/
struct RemoterPanel_Widgets {
    Fl_Choice*          robot_to_control;
    Fl_Value_Slider*    rc_throttle;
    Fl_Value_Slider*    rc_roll;
    Fl_Value_Slider*    rc_pitch;
    Fl_Value_Slider*    rc_yaw;
};
class RemoterPanel : public Fl_Window
{
public:
    RemoterPanel(int xpos, int ypos, int width, int height, const char* title);
    static Fl_Button*  remoter_button; // contain the handle of the button which open this panel in its callback
    // widgets
    struct RemoterPanel_Widgets ws;
private:
    // callback funcs
    static void cb_close(Fl_Widget*, void*);
    static void cb_change_robot_choice(Fl_Widget*, void*);
    static void cb_change_rc_throttle(Fl_Widget*, void*);
    static void cb_change_rc_roll(Fl_Widget*, void*);
    static void cb_change_rc_pitch(Fl_Widget*, void*);
    static void cb_change_rc_yaw(Fl_Widget*, void*);
    static void save_value_to_configs(RemoterPanel_Widgets*);
};
Fl_Button* RemoterPanel::remoter_button = NULL;
void RemoterPanel::cb_close(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        ((Fl_Window*)w)->hide(); 
        // and release the remote control button in the parent window
        if (remoter_button != NULL)
            remoter_button->value(0);
    }
}
void RemoterPanel::cb_change_robot_choice(Fl_Widget* w, void* data) {
}
void RemoterPanel::cb_change_rc_throttle(Fl_Widget* w, void* data) {

}
void RemoterPanel::cb_change_rc_roll(Fl_Widget* w, void* data) {
}
void RemoterPanel::cb_change_rc_pitch(Fl_Widget* w, void* data) {
}
void RemoterPanel::cb_change_rc_yaw(Fl_Widget* w, void* data) {
}
void save_value_to_configs(RemoterPanel_Widgets* ws) {
}
RemoterPanel::RemoterPanel(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    callback(cb_close);
    begin();
    int t_x = 5, t_y = 5, t_w = w()-10, t_h = h()-10;
    // choose robot
    ws.robot_to_control = new Fl_Choice(t_x+140, t_y, 60, 25, "Robot to control");
    const char* robot_name[] = {"#1", "#2", "#3", "#4"};
    for (char i; i < 4; i++) // 4 robots max
        ws.robot_to_control->add(robot_name[i]);
    ws.robot_to_control->value(0); // the first robot by default
    ws.robot_to_control->callback(cb_change_robot_choice, &ws);
    // RC control
    ws.rc_throttle = new Fl_Value_Slider(t_x+60, t_y+30, 220, 25, "Throttle");
    ws.rc_roll = new Fl_Value_Slider(t_x+60, t_y+60, 220, 25, "Roll");
    ws.rc_pitch = new Fl_Value_Slider(t_x+60, t_y+90, 220, 25, "Pitch");
    ws.rc_yaw = new Fl_Value_Slider(t_x+60, t_y+120, 220, 25, "Yaw");
    ws.rc_throttle->type(FL_HOR_FILL_SLIDER);
    ws.rc_roll->type(FL_HOR_FILL_SLIDER);
    ws.rc_pitch->type(FL_HOR_FILL_SLIDER);
    ws.rc_yaw->type(FL_HOR_FILL_SLIDER);
    ws.rc_throttle->selection_color(FL_DARK_GREEN);
    ws.rc_roll->selection_color(FL_DARK_RED);
    ws.rc_pitch->selection_color(FL_DARK_CYAN);
    ws.rc_yaw->selection_color(FL_DARK_MAGENTA);
    ws.rc_throttle->align(Fl_Align(FL_ALIGN_LEFT));
    ws.rc_roll->align(Fl_Align(FL_ALIGN_LEFT));
    ws.rc_pitch->align(Fl_Align(FL_ALIGN_LEFT));
    ws.rc_yaw->align(Fl_Align(FL_ALIGN_LEFT));
    ws.rc_throttle->range(1000, 2000);
    ws.rc_roll->range(1000, 2000);
    ws.rc_pitch->range(1000, 2000);
    ws.rc_yaw->range(1000, 2000);
    ws.rc_throttle->step(1);
    ws.rc_roll->step(1);
    ws.rc_pitch->step(1);
    ws.rc_yaw->step(1);
    ws.rc_throttle->callback(cb_change_rc_throttle);
    ws.rc_roll->callback(cb_change_rc_roll);
    ws.rc_pitch->callback(cb_change_rc_pitch);
    ws.rc_yaw->callback(cb_change_rc_yaw);
    end();
    show();
}

/*------- Robot panel (state viewer & controller) -------*/
struct RobotPanel_Widgets { // for parameter saving
    Fl_LED_Button*  robot_link_state[4]; // 4 robots max
    Fl_Box*         robot_arm_state[4]; // 4 robots max
    Fl_Button*      robot_rc_button;
};
struct RobotPanel_handles {
    RemoterPanel*   remoter_panel;
};
class RobotPanel : public Fl_Window
{
public:
    RobotPanel(int xpos, int ypos, int width, int height, const char* title);
    static Fl_Button*  robot_button; // contain the handle of the button which open this panel in its callback
    // widgets
    struct RobotPanel_Widgets ws;
    static struct RobotPanel_handles hs;
private:
    // callback funcs
    static void cb_close(Fl_Widget*, void*);
    static void cb_robot_rc_button(Fl_Widget*, void*);
    // function to save current value of widgets to runtime configs
    static void save_value_to_configs(RobotPanel_Widgets*);
    // function to get runtime configs to set value of widgets
    static void get_value_from_configs(RobotPanel_Widgets*);
};
Fl_Button* RobotPanel::robot_button = NULL;
struct RobotPanel_handles RobotPanel::hs = {NULL};
void RobotPanel::cb_close(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        ((Fl_Window*)w)->hide();
        // and release the robot button in toolbar
        if (robot_button != NULL)
            robot_button->value(0);
    }
}
void RobotPanel::cb_robot_rc_button(Fl_Widget* w, void* data) {
    if (hs.remoter_panel != NULL) {
        if (hs.remoter_panel->shown()) {
            if (!((Fl_Button*)w)->value()) {
                hs.remoter_panel->hide();
            }
        }
        else {
            if (((Fl_Button*)w)->value()) {
                hs.remoter_panel->show();
            }
        }
    }
    else // first press this button
    {// create config dialog
        if (((Fl_Button*)w)->value()) // if pressed
        {
            Fl_Window* window = w->window(); // find the nearest parent window of this button, i.e., RobotPanel
            hs.remoter_panel = new RemoterPanel(window->x()+window->w(), window->y(), 
                300, window->h(), "RC Control");
            hs.remoter_panel->remoter_button = (Fl_Button*)w;
        }
    }
}
void RobotPanel::get_value_from_configs(RobotPanel_Widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
    
    // check whether to open remoter panel or not
    if (configs->system.remoter_panel_opened) {
        // open remoter panel
        Fl_Window* window = ws->robot_rc_button->window();
        hs.remoter_panel = new RemoterPanel(window->x()+window->w(), window->y(), 
                300, window->h(), "RC Control");
        hs.remoter_panel->remoter_button = ws->robot_rc_button;
        ws->robot_rc_button->value(1);
    }
}
RobotPanel::RobotPanel(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs

    // add event handle to dialog window
    callback(cb_close, (void*)&ws);   
    // begin adding children
    begin();
    int t_x = 5, t_y = 5, t_w = w()-10, t_h = h()-10;
    //  robot link state, Note: only check data network (data receiving)
    Fl_Box *link = new Fl_Box(t_x, t_y, 160, 160, "Robot State");
        link->box(FL_PLASTIC_UP_FRAME);
        link->labelsize(15);
        link->labelfont(FL_COURIER_BOLD_ITALIC);
        link->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
    {
        // LED indicating data link state
        new Fl_Box(t_x, t_y+20, 80, 25, "Data Link");
        const char* robot_name[] = {"#1", "#2", "#3", "#4"};
        for (char i = 0; i < 4; i++) // 4 robots max
        {
            ws.robot_link_state[i] = new Fl_LED_Button(t_x+30, t_y+40+30*i, 30, 30, robot_name[i]);
            ws.robot_link_state[i]->selection_color(FL_DARK_GREEN);
            ws.robot_link_state[i]->labelsize(13);
            ws.robot_link_state[i]->align(Fl_Align(FL_ALIGN_LEFT)); 
        }
        // ARM/DISARM info
        new Fl_Box(t_x+90, t_y+20, 60, 25, "ARMING");
        for (char i = 0; i < 4; i++) // 4 robots max
        {
            ws.robot_arm_state[i] = new Fl_Box(t_x+90, t_y+40+30*i, 60, 25, "DISARM");
            ws.robot_arm_state[i]->labelcolor(FL_RED);
        }
    }
    //  robot remote control
    ws.robot_rc_button = new Fl_Button(t_x, t_y+40+30*4, 34, 34);
    Fl_Pixmap *icon_rc = new Fl_Pixmap(pixmap_icon_rc);
    ws.robot_rc_button->image(icon_rc);
    ws.robot_rc_button->tooltip("Robot remote controller, please use with care.");
    ws.robot_rc_button->type(FL_TOGGLE_BUTTON);
    ws.robot_rc_button->callback(cb_robot_rc_button);
    new Fl_Box(t_x+40, t_y+40+30*4, 120, 30, "Remote Control");
    end();

    // set values from configs
    get_value_from_configs(&ws);

    show();
}

/*------- Result panel -------*/
struct ResultPanel_Widgets { // for parameter saving
};
struct ResultPanel_handles {
};
class ResultPanel : public Fl_Window
{
public:
    ResultPanel(int xpos, int ypos, int width, int height, const char* title);
    static Fl_Button*  result_button; // contain the handle of the button which open this panel in its callback
    // widgets
    struct ResultPanel_Widgets ws;
    static struct ResultPanel_handles hs;
private:
    // callback funcs
    static void cb_close(Fl_Widget*, void*);
    // function to save current value of widgets to runtime configs
    static void save_value_to_configs(ResultPanel_Widgets*);
    // function to get runtime configs to set value of widgets
    static void get_value_from_configs(ResultPanel_Widgets*);
};
Fl_Button* ResultPanel::result_button = NULL;
//struct ResultPanel_handles ResultPanel::hs = {NULL};
void ResultPanel::cb_close(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        ((Fl_Window*)w)->hide();
        // and release the result button in toolbar
        if (result_button != NULL)
            result_button->value(0);
    }
}
void ResultPanel::get_value_from_configs(ResultPanel_Widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
}
ResultPanel::ResultPanel(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs

    // add event handle to dialog window
    callback(cb_close, (void*)&ws);   
    // begin adding children
    begin();
    int t_x = 5, t_y = 5, t_w = w()-10, t_h = h()-10;
    
    end();

    // set values from configs
    get_value_from_configs(&ws);

    show();
}

/*------- ToolBar -------*/
struct ToolBar_Widgets
{
    Fl_Button* start; // start button
    Fl_Button* pause; // pause button
    Fl_Button* stop; // stop button
    Fl_Button* config; // config button
    Fl_Light_Button* record; // record button
    Fl_Button* robot; // robot state&control button
    Fl_Button* result; // result display button
};
struct ToolBar_Handles // handles of dialogs/panels opened by corresponding buttons
{
    ConfigDlg* config_dlg; // handle of config dialog opened by config button
    RobotPanel* robot_panel; // handle of robot panel opened by robot button
    ResultPanel* result_panel; // handle of result panel opened by result button
};
class ToolBar : public Fl_Group
{
public:
    ToolBar(int Xpos, int Ypos, int Width, int Height, void *win);
    struct ToolBar_Widgets ws;
    static struct ToolBar_Handles hs;
    void restore_from_configs(ToolBar_Widgets*, void*);
private:
    static void cb_button_start(Fl_Widget*, void*);
    static void cb_button_pause(Fl_Widget*, void*);
    static void cb_button_stop(Fl_Widget*, void*);
    static void cb_button_config(Fl_Widget*, void*);
    static void cb_button_robot(Fl_Widget*, void*);
    static void cb_button_result(Fl_Widget*, void*); 
};
struct ToolBar_Handles ToolBar::hs = {NULL, NULL, NULL};

void ToolBar::cb_button_start(Fl_Widget *w, void *data)
{
    fl_alert("Start Button pressed!");
}

void ToolBar::cb_button_pause(Fl_Widget *w, void *data)
{
    fl_alert("Pause Button pressed!");
}

void ToolBar::cb_button_stop(Fl_Widget *w, void *data)
{
    struct ToolBar_Widgets *widgets = (struct ToolBar_Widgets*)data;
    widgets->start->clear();
    widgets->pause->clear();
}

void ToolBar::cb_button_config(Fl_Widget *w, void *data)
{
    
    if (hs.config_dlg != NULL)
    {
        if (hs.config_dlg->shown()) // if shown, do not open again
        {}
        else
            hs.config_dlg->show();
    }
    else // first press this button
    {// create config dialog
        Fl_Window* window=(Fl_Window*)data;
        hs.config_dlg = new ConfigDlg(window->x()+20, window->y()+20, 
            400, 400, "Settings");
    }
}

void ToolBar::cb_button_robot(Fl_Widget *w, void *data)
{
    if (hs.robot_panel != NULL)
    {
        if (hs.robot_panel->shown()) { // if shown, do not open again
            if (!((Fl_Button*)w)->value())
                hs.robot_panel->hide();
        }
        else {
            if (((Fl_Button*)w)->value())
                hs.robot_panel->show();
        }
    }
    else // first press this button
    {// create config dialog
        if (((Fl_Button*)w)->value()) // if pressed
        {
            Fl_Window* window=(Fl_Window*)data;
            hs.robot_panel = new RobotPanel(window->x(), window->y()+window->h()+40, 
                window->w(), 200, "Robot Panel");
            hs.robot_panel->robot_button = (Fl_Button*)w;
        }
    }
}

void ToolBar::cb_button_result(Fl_Widget *w, void *data)
{
    if (hs.result_panel != NULL)
    {
        if (hs.result_panel->shown()) { // if shown, do not open again
            if (!((Fl_Button*)w)->value())
                hs.result_panel->hide();
        }
        else {
            if (((Fl_Button*)w)->value())
                hs.result_panel->show();
        }
    }
    else // first press this button
    {// create config dialog
        if (((Fl_Button*)w)->value()) // if pressed
        {
            Fl_Window* window=(Fl_Window*)data;
            hs.result_panel = new ResultPanel(window->x()+window->w(), window->y(), 
                200, window->h(), "Result Panel");
            hs.result_panel->result_button = (Fl_Button*)w;
        }
    }
}
void ToolBar::restore_from_configs(ToolBar_Widgets* ws, void *data)
{
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs

    // check whether to open robot panel or not
    if (configs->system.robot_panel_opened) {
        // open remoter panel
        Fl_Window* window = (Fl_Window*)data;
        hs.robot_panel = new RobotPanel(window->x(), window->y()+window->h()+40, 
            window->w(), 200, "Robot Panel");
        hs.robot_panel->robot_button = ws->robot;
        ws->robot->value(1);
    }
    // check whether to open result panel or not
    if (configs->system.result_panel_opened) {
        // open remoter panel
        Fl_Window* window = (Fl_Window*)data;
        hs.result_panel = new ResultPanel(window->x()+window->w(), window->y(), 
                200, window->h(), "Result Panel");
        hs.result_panel->result_button = ws->result;
        ws->result->value(1);
    }
}
ToolBar::ToolBar(int Xpos, int Ypos, int Width, int Height, void *win) :
Fl_Group(Xpos, Ypos, Width, Height)
{
    begin();
    Fl_Box *bar = new Fl_Box(FL_UP_BOX, 0, 0, Width, Height, "");
    Ypos += 2; Height -= 4; Xpos += 3; Width = Height;
    // widgets of this toolbar
    //struct ToolBar_Widgets ws;
    // instances of buttons belong to tool bar
    ws.start = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    ws.pause = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    ws.stop = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    ws.config = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    ws.record = new Fl_Light_Button(Xpos, Ypos, Width+22, Height); Xpos += Width+22+5;
    ws.robot = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    ws.result = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    Fl_Box *bar_rest = new Fl_Box(FL_DOWN_BOX, Xpos, Ypos, bar->w()-Xpos, Height, "");
    resizable(bar_rest); // protect buttons from resizing
    // icons
    Fl_Pixmap *icon_start = new Fl_Pixmap(pixmap_icon_play);
    Fl_Pixmap *icon_pause = new Fl_Pixmap(pixmap_icon_pause);
    Fl_Pixmap *icon_stop = new Fl_Pixmap(pixmap_icon_stop);
    Fl_Pixmap *icon_config = new Fl_Pixmap(pixmap_icon_config);
    Fl_Pixmap *icon_record = new Fl_Pixmap(pixmap_icon_record);
    Fl_Pixmap *icon_robot = new Fl_Pixmap(pixmap_icon_helicopter);
    Fl_Pixmap *icon_result = new Fl_Pixmap(pixmap_icon_result);
    // link icons to buttons
    ws.start->image(icon_start);
    ws.pause->image(icon_pause);
    ws.stop->image(icon_stop);
    ws.config->image(icon_config);
    ws.record->image(icon_record);
    ws.robot->image(icon_robot);
    ws.result->image(icon_result);
    // tips for buttons
    ws.start->tooltip("Start Searching");
    ws.pause->tooltip("Pause Searching");
    ws.stop->tooltip("Stop Searching");
    ws.config->tooltip("Settings");
    ws.record->tooltip("Recording");
    ws.robot->tooltip("Robot viewer & controller");
    ws.result->tooltip("Result viewer");
    // types of buttons
    ws.start->type(FL_RADIO_BUTTON); // start & pause are mutually exclusive
    ws.pause->type(FL_RADIO_BUTTON);
    ws.robot->type(FL_TOGGLE_BUTTON);
    ws.result->type(FL_TOGGLE_BUTTON);
    // colors
    ws.record->selection_color(FL_RED);
    // link call backs to buttons
    ws.start->callback(cb_button_start);
    ws.pause->callback(cb_button_pause);
    //  start & pause buttons will be released when stop button is pressed
    ws.stop->callback(cb_button_stop, (void*)&ws);
    //  config dialog will pop up when config button pressed
    ws.config->callback(cb_button_config, (void*)win);
    //  robot window will pop up when robot button pressed
    ws.robot->callback(cb_button_robot, (void*)win);
    //  result window will pop up when result button pressed
    ws.result->callback(cb_button_result, (void*)win);
    end();
}


/*------- Creation function of User Interface  -------*/
void UI::cb_close(Fl_Widget* w, void* data) { 
    // close GSRAO
    if (Fl::event() == FL_CLOSE) {
        // save open/close states of other sub-panels to configs
        GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
        UI_Widgets* ws = (UI_Widgets*)data;
        if (((ToolBar*)ws->toolbar)->hs.robot_panel != NULL) { // robot panel
            if (((ToolBar*)ws->toolbar)->hs.robot_panel->shown())
                configs->system.robot_panel_opened = true;
            else
                configs->system.robot_panel_opened = false;
            if (((ToolBar*)ws->toolbar)->hs.robot_panel->hs.remoter_panel != NULL) { // robot panel -> remoter control panel
                if (((ToolBar*)ws->toolbar)->hs.robot_panel->hs.remoter_panel->shown())
                    configs->system.remoter_panel_opened = true;
                else
                    configs->system.remoter_panel_opened = false;
            }
        }
        if (((ToolBar*)ws->toolbar)->hs.result_panel != NULL) { // result panel
            if (((ToolBar*)ws->toolbar)->hs.result_panel->shown())
                configs->system.result_panel_opened = true;
            else
                configs->system.result_panel_opened = false;
        }
        // close other panels
        if (((ToolBar*)ws->toolbar)->hs.robot_panel->hs.remoter_panel != NULL && ((ToolBar*)ws->toolbar)->hs.robot_panel->hs.remoter_panel->shown()) // close remoter panel
            ((ToolBar*)ws->toolbar)->hs.robot_panel->hs.remoter_panel->hide();
        if (((ToolBar*)ws->toolbar)->hs.robot_panel != NULL && ((ToolBar*)ws->toolbar)->hs.robot_panel->shown()) // close robot panel
            ((ToolBar*)ws->toolbar)->hs.robot_panel->hide();
        if (((ToolBar*)ws->toolbar)->hs.result_panel != NULL && ((ToolBar*)ws->toolbar)->hs.result_panel->shown()) // close result panel
            ((ToolBar*)ws->toolbar)->hs.result_panel->hide();

        // close main window
        ((Fl_Window*)w)->hide();
    }
}
UI::UI(int width, int height, const char* title=0)
{
    /* Main Window, control panel */
    Fl_Double_Window *ui = new Fl_Double_Window(0, 0, width, height, title);
    ui->resizable(ui); 
 
    ui->show(); // glut will die unless parent window visible
    /* begin adding children */
    ui->begin();
    // Add tool bar, it's width is equal to panel's
    ToolBar* tool = new ToolBar(0, 0, width, 34, (void*)ui);
    ws.toolbar = tool;
    tool->clear_visible_focus(); //just use mouse, no TABs
    // protect buttons from resizing
    Fl_Box *r = new Fl_Box(FL_NO_BOX, width, tool->h(), 0, height-tool->h(), "right_border");
    r->hide();
    ui->resizable(r);
    /* Add RAO view */    
    glutInitWindowSize(width-10, height-tool->h()-10);// be consistent with View_init
    glutInitWindowPosition(ui->x()+5, tool->h()+5); // place it inside parent window
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutCreateWindow("Experiment view");
    /* end adding children */
    ui->end();
    ui->resizable(glut_window);
    ui->callback(cb_close, &ws);// callback
 
    // init view
    View_init(width-10, height-tool->h()-10);// pass gl window size

    // open panels according to last use info
    tool->restore_from_configs(&(tool->ws), (void*)ui);
};
/* End of UI.cxx */
