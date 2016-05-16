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
#include "GSRAO_Config.h"
/* Linux Network */
#include <ifaddrs.h>
#include <sys/socket.h>
#include <arpa/inet.h>

/*------- Configuration Dialog -------*/
struct ConfigDlg_widgets { // for parameter saving
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
private:
    // widgets
    struct ConfigDlg_widgets dlg_w;
    // callback funcs
    static void cb_dlg(Fl_Widget*, void*);
    static void cb_switch_tabs(Fl_Widget*, void*);
    static void cb_change_num_of_robots(Fl_Widget*, void*);
    // function to save current value of widgets to runtime configs
    static void save_value_to_configs(ConfigDlg_widgets*);
    // function to get runtime configs to set value of widgets
    static void set_value_from_configs(ConfigDlg_widgets*);
};

void ConfigDlg::cb_dlg(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        struct ConfigDlg_widgets *ws = (struct ConfigDlg_widgets*)data;
        // save widget values to GSRAO runtime configs when closing the dialog window
        save_value_to_configs(ws);
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
    struct ConfigDlg_widgets *ws = (struct ConfigDlg_widgets*)data;

    // deactivate & activate corresponding mocap rigid body selections
    for (char i = ws->scenario_num_of_robots->value()+1; i < 4; i++) // 4 robots max
        ws->mocap_rigid_body_num_of_robot[i]->deactivate();
    for (char i = 0; i <= ws->scenario_num_of_robots->value(); i++)
        ws->mocap_rigid_body_num_of_robot[i]->activate();
}

void ConfigDlg::save_value_to_configs(ConfigDlg_widgets* ws) {
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

void ConfigDlg::set_value_from_configs(ConfigDlg_widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
}

ConfigDlg::ConfigDlg(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs

    // add event handle to dialog window
    callback(cb_dlg, (void*)&dlg_w);   
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
            scenario->color(0xe8e8e800); // light milk tea
            scenario->selection_color(0xe8e8e800); // light milk tea

            // number of robots
            dlg_w.scenario_num_of_robots = new Fl_Choice(t_x+10+160, t_y+25+10, 100, 25,"Number of robots ");
            dlg_w.scenario_num_of_robots->add("1");
            dlg_w.scenario_num_of_robots->add("2");
            dlg_w.scenario_num_of_robots->add("3");
            dlg_w.scenario_num_of_robots->add("4");
            dlg_w.scenario_num_of_robots->value(configs->robot.num_of_robots-1); // Fl_Choice count from 0
            dlg_w.scenario_num_of_robots->callback(cb_change_num_of_robots, (void*)&dlg_w);
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
            dlg_w.ppmcnt_serial_port = new Fl_Input(t_x+10+100, t_y+25+10+30, 200, 25, "Serial Port ");
            dlg_w.ppmcnt_serial_port->value(configs->robot.ppm_serial_port_path.c_str());

            // Data network
            Fl_Box *dnet = new Fl_Box(t_x+10, t_y+25+10+70, 370, 65,"Data Network");
            dnet->box(FL_PLASTIC_UP_FRAME);
            dnet->labelsize(16);
            dnet->labelfont(FL_COURIER_BOLD_ITALIC);
            dnet->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            //   Set serial port receiving the data
            dlg_w.dnet_serial_port = new Fl_Input(t_x+10+100, t_y+25+10+100, 200, 25, "Serial Port ");
            dlg_w.dnet_serial_port->value(configs->robot.dnet_serial_port_path.c_str());

            // Motion capture settings
            Fl_Box *mocap = new Fl_Box(t_x+10, t_y+25+10+140, 370, 130,"Motion Capture");
            mocap->box(FL_PLASTIC_UP_FRAME);
            mocap->labelsize(16);
            mocap->labelfont(FL_COURIER_BOLD_ITALIC);
            mocap->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            //   Select network interface receiving the multicast info of Motion Capture System
            dlg_w.mocap_netcard = new Fl_Choice(t_x+10+70, t_y+25+10+30+140, 280, 25, "Netcard");
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
                    dlg_w.mocap_netcard->add(ncName);
                    // count netcard number
                    netcard_count++;
                } else if (ifa->ifa_addr->sa_family == AF_INET6) { // check it is IP6
                    // is a valid IP6 Address
                    tmpAddrPtr=&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
                    char addressBuffer[INET6_ADDRSTRLEN];
                    inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
                    // add this net interface to choice list
                    snprintf(ncName, 100, "%s %s IPv6", ifa->ifa_name, addressBuffer);
                    dlg_w.mocap_netcard->add(ncName);
                    // count netcard number
                    netcard_count++;
                } 
            }
            if (ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);
            //    default choice
            if (configs->mocap.netcard < netcard_count)
                dlg_w.mocap_netcard->value(configs->mocap.netcard);
            else
                dlg_w.mocap_netcard->value(0);
            dlg_w.mocap_netcard->tooltip("Select which network interface receives multicast info from Motive software");

            //   Config rigid body index for microdrones
            const char* robot_name[] = {"robot 1", "robot 2", "robot 3", "robot 4"};
            char rbName[20];
            for (char i = 0; i < 4; i++) // 4 robots max
            {
                dlg_w.mocap_rigid_body_num_of_robot[i] = new Fl_Choice(t_x+10+70+175*(i%2), t_y+25+10+60+30*(i<2?0:1)+140, 120, 25, robot_name[i]);
                for (char j = 0; j < 10; j++) // 10 rigid body candidates
                {
                    snprintf(rbName, 20, "rigid body %d", j+1);
                    dlg_w.mocap_rigid_body_num_of_robot[i]->add(rbName);
                    dlg_w.mocap_rigid_body_num_of_robot[i]->tooltip("Select corresponding rigid body number of the robot");
                }
                // set choice according to configs
                dlg_w.mocap_rigid_body_num_of_robot[i]->value(configs->mocap.rigid_body_num_of_robot[i]);
                // activate/deactivate according to number of robots
                if (i <= dlg_w.scenario_num_of_robots->value())
                    dlg_w.mocap_rigid_body_num_of_robot[i]->activate();
                else
                    dlg_w.mocap_rigid_body_num_of_robot[i]->deactivate();
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
    // set value according to runtime configs
    set_value_from_configs(&dlg_w);
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
};
class ToolBar : public Fl_Group
{
public:
    ToolBar(int Xpos, int Ypos, int Width, int Height, void *win);
    struct ToolBar_Widgets tb_widgets;
    static void cb_button_start(Fl_Widget*, void*);
    static void cb_button_pause(Fl_Widget*, void*);
    static void cb_button_stop(Fl_Widget*, void*);
    static void cb_button_config(Fl_Widget*, void*);
};

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
    // Open Configuration dialog
    Fl_Window* window=(Fl_Window*)data;
    ConfigDlg *config = new ConfigDlg(window->x()+20, window->y()+20, 
            400, 400, "Settings");
}

ToolBar::ToolBar(int Xpos, int Ypos, int Width, int Height, void *win) :
Fl_Group(Xpos, Ypos, Width, Height)
{
    begin();
    Fl_Box *bar = new Fl_Box(FL_UP_BOX, 0, 0, Width, Height, "");
    Ypos += 2; Height -= 4; Xpos += 3; Width = Height;
    // widgets of this toolbar
    //struct ToolBar_Widgets tb_widgets;
    // instances of buttons belong to tool bar
    tb_widgets.start = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    tb_widgets.pause = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    tb_widgets.stop = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    tb_widgets.config = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width + 5;
    tb_widgets.record = new Fl_Light_Button(Xpos, Ypos, Width+22, Height); Xpos += Width+22+5;
    Fl_Box *bar_rest = new Fl_Box(FL_DOWN_BOX, Xpos, Ypos, bar->w()-Xpos, Height, "");
    resizable(bar_rest); // protect buttons from resizing
    // icons
    Fl_Pixmap *icon_start = new Fl_Pixmap(pixmap_icon_play);
    Fl_Pixmap *icon_pause = new Fl_Pixmap(pixmap_icon_pause);
    Fl_Pixmap *icon_stop = new Fl_Pixmap(pixmap_icon_stop);
    Fl_Pixmap *icon_config = new Fl_Pixmap(pixmap_icon_config);
    Fl_Pixmap *icon_record = new Fl_Pixmap(pixmap_icon_record);
    // link icons to buttons
    tb_widgets.start->image(icon_start);
    tb_widgets.pause->image(icon_pause);
    tb_widgets.stop->image(icon_stop);
    tb_widgets.config->image(icon_config);
    tb_widgets.record->image(icon_record);
    // tips for buttons
    tb_widgets.start->tooltip("Start Simulation");
    tb_widgets.pause->tooltip("Pause Simulation");
    tb_widgets.stop->tooltip("Stop Simulation");
    tb_widgets.config->tooltip("Settings");
    tb_widgets.record->tooltip("Recording");
    // types of buttons
    tb_widgets.start->type(FL_RADIO_BUTTON); // start & pause are mutually exclusive
    tb_widgets.pause->type(FL_RADIO_BUTTON);
    // colors
    tb_widgets.record->selection_color(FL_RED);
    // link call backs to buttons
    tb_widgets.start->callback(cb_button_start);
    tb_widgets.pause->callback(cb_button_pause);
    //  start & pause buttons will be released when stop button is pressed
    tb_widgets.stop->callback(cb_button_stop, (void*)&tb_widgets);
    tb_widgets.config->callback(cb_button_config, (void*)win);
    end();
}


/*------- Creation function of User Interface  -------*/
UI::UI(int width, int height, const char* title=0)
{
    /* Main Window, control panel */
    Fl_Double_Window *panel = new Fl_Double_Window(width, height, title);
    panel->resizable(panel);
   
    // Add tool bar, it's width is equal to panel's
    ToolBar *tool = new ToolBar(0, 0, width, 34, (void*)panel);
    tool->clear_visible_focus(); //just use mouse, no TABs
    // protect buttons from resizing
    Fl_Box *r = new Fl_Box(FL_NO_BOX, width, tool->h(), 0, height-tool->h(), "right_border");
    r->hide();
    panel->resizable(r);

    /* Add RAO view */
    panel->show(); // glut will die unless parent window visible
    /* begin adding children */
    panel->begin(); 
    glutInitWindowSize(width-10, height-tool->h()-10);// be consistent with View_init
    glutInitWindowPosition(panel->x()+5, tool->h()+5); // place it inside parent window
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutCreateWindow("Experiment view");
    /* end adding children */
    panel->end();
    panel->resizable(glut_window); 
 
    // init view
    View_init(width-10, height-tool->h()-10);// pass gl window size
};
/* End of UI.cxx */
