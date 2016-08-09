/*
 * User Interface of RAO Player
 *         using FLTK
 *
 * Author: Roice (LUO Bing)
 * Date: 2016-08-07 create this file
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
#include <FL/Fl_Scroll.H>
#include <FL/Fl_File_Chooser.H>
/* OpenGL */
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <FL/glut.H>
/* RAO Player */
#include "ui/UI.h"
#include "ui/icons/icons.h" // pixmap icons used in Tool bar
#include "ui/View.h" // 3D RAO view
#include "ui/widgets/Fl_LED_Button/Fl_LED_Button.H"
#include "ui/draw/draw_wave.h"
#include "io/record.h"
#include "Player_Config.h"

/*------- Configuration Dialog -------*/
struct ConfigDlg_Widgets { // for parameter saving
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

void ConfigDlg::save_value_to_configs(ConfigDlg_Widgets* ws) {
    Config_t* configs = Config_get_configs(); // get runtime configs
}

void ConfigDlg::get_value_from_configs(ConfigDlg_Widgets* ws) {
    Config_t* configs = Config_get_configs(); // get runtime configs
}

ConfigDlg::ConfigDlg(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{

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
        }
        scenario->end();

        // Tab Link
        Fl_Group *link = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Link");
        {
            // color of this tab
            link->color(0xe8e8e800); // light milk tea
            link->selection_color(0xe8e8e800); // light milk tea
        }
        link->end();

        // Tab Flow
        Fl_Group *flow = new Fl_Group(t_x,t_y+25,t_w,t_h-25,"Flow");
        {
            // color of this tab
            flow->color(0xe0ffff00); // light blue
            flow->selection_color(0xe0ffff00); // light blue
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

/* =================================================
 * ==== Robot panel (state viewer & controller) ====
 * =================================================*/
struct RobotPanel_Widgets { // for parameter saving
    Fl_LED_Button*  robot_link_state[4]; // 4 robots max
    Fl_Box*         robot_arm_state[4]; // 4 robots max
    Fl_Box*         robot_bat_state[4]; // 4 robots max
    Fl_Choice*      robot_to_display_sensor_reading; // choose which robot's reading to display
    WavePlot*      robot_sensor_reading; // reading of sensors
};
struct RobotPanel_handles {
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
    // function to save current value of widgets to runtime configs
    static void save_value_to_configs(RobotPanel_Widgets*);
    // function to get runtime configs to set value of widgets
    static void get_value_from_configs(RobotPanel_Widgets*);
};
Fl_Button* RobotPanel::robot_button = NULL;
//struct RobotPanel_handles RobotPanel::hs = {NULL};
void RobotPanel::cb_close(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        ((Fl_Window*)w)->hide();
        // and release the robot button in toolbar
        if (robot_button != NULL)
            robot_button->value(0);
    }
}

void RobotPanel::get_value_from_configs(RobotPanel_Widgets* ws) {
    Config_t* configs = Config_get_configs(); // get runtime configs
}

RobotPanel::RobotPanel(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    Config_t* configs = Config_get_configs(); // get runtime configs

    // add event handle to dialog window
    callback(cb_close, (void*)&ws);   
    // begin adding children
    begin();
    int t_x = 5, t_y = 5, t_w = w()-10, t_h = h()-10;
    //  robot link state, Note: only check data network (data receiving)
    Fl_Box *link = new Fl_Box(t_x, t_y, 220, 160, "Robot State");
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
        // Battery status
        new Fl_Box(t_x+160, t_y+20, 60, 25, "Battery");
        for (char i = 0; i < 4; i++) // 4 robots max
        {
            ws.robot_bat_state[i] = new Fl_Box(t_x+150, t_y+40+30*i, 60, 25, "0 V");
            ws.robot_bat_state[i]->labelcolor(FL_RED);
        }
    }

    // robot choice to display sensor reading
    ws.robot_to_display_sensor_reading = new Fl_Choice(t_x+40, t_y+40+30*4+2, 180, 30);
    ws.robot_to_display_sensor_reading->add("Show sensors robot 1");
    ws.robot_to_display_sensor_reading->add("Show sensors robot 2");
    ws.robot_to_display_sensor_reading->add("Show sensors robot 3");
    ws.robot_to_display_sensor_reading->add("Show sensors robot 4");
    ws.robot_to_display_sensor_reading->value(0);

    // sensor reading plot
    Fl_Box* sr_box = new Fl_Box(t_x+225, t_y, 465, 190, "Sensor reading");
    sr_box->box(FL_PLASTIC_UP_FRAME);
    sr_box->labelsize(15);
    sr_box->labelfont(FL_COURIER_BOLD_ITALIC);
    sr_box->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
    {
        Fl_Scroll* scroll = new Fl_Scroll(t_x+230, t_y+25, 455, 165);
        ws.robot_sensor_reading = new WavePlot(t_x+230, t_y+25, 455*10, 140, ""); // *10 means 10 min length of data 
        scroll->end();
    }
    end();

    // set values from configs
    get_value_from_configs(&ws);

    show();
}

/* ================================
 * ========= Result Panel =========
 * ================================*/
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
    Config_t* configs = Config_get_configs(); // get runtime configs
}
ResultPanel::ResultPanel(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
    Config_t* configs = Config_get_configs(); // get runtime configs

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

/* ================================
 * ========= ToolBar ==============
 * ================================*/
struct ToolBar_Widgets
{
    Fl_Button*          start;  // start button
    Fl_Button*          pause;  // pause button
    Fl_Button*          stop;   // stop button
    Fl_Button*          config; // config button
    Fl_Light_Button*    record; // record button
    Fl_Button*          robot;  // robot state&control button
    Fl_Button*          result; // result display button
    Fl_Button*          open;   // choose file
    Fl_Box*             msg_zone; // message zone
};
struct ToolBar_Handles // handles of dialogs/panels opened by corresponding buttons
{
    ConfigDlg* config_dlg; // handle of config dialog opened by config button
    RobotPanel* robot_panel; // handle of robot panel opened by robot button
    ResultPanel* result_panel; // handle of result panel opened by result button
    Fl_File_Chooser* fc; // handle of file chooser opened by open button
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
    static void cb_button_open(Fl_Widget*, void*);
};
struct ToolBar_Handles ToolBar::hs = {NULL, NULL, NULL, NULL};

/*------- Repeated Tasks -------*/
static void cb_repeated_tasks_2hz(void* data)
{
#if 0
    ToolBar_Handles* hs = (ToolBar_Handles*)data;
    // refresh robots' states in robot panel
    if (hs->robot_panel != NULL)
    {
        if (hs->robot_panel->shown())
        {
            MicroBee_t* mb = microbee_get_states();
            char label_name[100];
            for (int i = 0; i < 4; i++)
            {
                if (mb[i].state.linked)
                    hs->robot_panel->ws.robot_link_state[i]->value(1);
                else
                    hs->robot_panel->ws.robot_link_state[i]->value(0);
                if (mb[i].state.armed)
                {
                    hs->robot_panel->ws.robot_arm_state[i]->label("ARM"); // arm/disarm
                    hs->robot_panel->ws.robot_arm_state[i]->labelcolor(FL_GREEN);
                }
                else
                {
                    hs->robot_panel->ws.robot_arm_state[i]->label("DISARM"); // arm/disarm
                    hs->robot_panel->ws.robot_arm_state[i]->labelcolor(FL_RED);
                }
                snprintf(label_name, 100, "%1.2f V", mb[i].state.bat_volt);
                hs->robot_panel->ws.robot_bat_state[i]->copy_label(label_name); // battery status
                hs->robot_panel->ws.robot_bat_state[i]->labelcolor(FL_BLUE);
            }
        }
    }
    // reload
    Fl::repeat_timeout(0.5, cb_repeated_tasks_2hz, data);
#endif
}

static void cb_repeated_tasks_10hz(void* data)
{
    ToolBar_Handles* hs = (ToolBar_Handles*)data;

    // draw sensor reading
    if (hs->robot_panel != NULL && hs->robot_panel->shown())
    {
        hs->robot_panel->ws.robot_sensor_reading->redraw();
    }

    // reload
    Fl::repeat_timeout(0.1, cb_repeated_tasks_10hz, data);
}

void ToolBar::cb_button_start(Fl_Widget *w, void *data)
{
    Config_t* configs = Config_get_configs(); // get runtime configs

    ToolBar_Widgets* widgets = (ToolBar_Widgets*)data;

    // if pause button is pressed, meaning that the initialization has been carried out, so just restore and continue
    if (widgets->pause->value()) {
        // release pause button
        widgets->pause->activate(); widgets->pause->clear();
        // continue running
        
    }
    else {
    // if pause button is not pressed, then need check start button state
        if (((Fl_Button*)w)->value()) // if start button is pressed down
        {
            // lock config button
            widgets->config->deactivate();
            widgets->msg_zone->label(""); // clear message zone
         
            // add timers for repeated tasks (such as data display)
            //Fl::add_timeout(0.5, cb_repeated_tasks_2hz, (void*)&hs);
            Fl::add_timeout(0.1, cb_repeated_tasks_10hz, (void*)&hs);
        }
        else {
            // user is trying to release start button when pause is not pressed
            ((Fl_Button*)w)->value(1);
        }
    }
}

void ToolBar::cb_button_pause(Fl_Widget *w, void *data)
{
    ToolBar_Widgets* widgets = (ToolBar_Widgets*)data;
    // if start button pressed, release it, and pause experiment
    if (widgets->start->value()) {
        widgets->start->value(0); // release start button
        widgets->pause->deactivate(); // make pause button unclickable
        // pause experiment...

    }
    else {
    // if start button not pressed, pause button will not toggle and no code action will be took
        widgets->pause->clear();
    }
}

void ToolBar::cb_button_stop(Fl_Widget *w, void *data)
{
    // release start and pause buttons
    struct ToolBar_Widgets *widgets = (struct ToolBar_Widgets*)data;
    widgets->start->clear();
    widgets->pause->activate(); widgets->pause->clear();

    //Fl::remove_timeout(cb_repeated_tasks_2hz); // remove timeout callback for repeated tasks
    Fl::remove_timeout(cb_repeated_tasks_10hz); // remove timeout callback for repeated tasks
    if (hs.robot_panel != NULL) {
        for (int i = 0; i < 4; i++) // clear robot states in robot panel
        {
            hs.robot_panel->ws.robot_link_state[i]->value(0); // linked leds
            hs.robot_panel->ws.robot_arm_state[i]->label("DISARM"); // arm/disarm
            hs.robot_panel->ws.robot_arm_state[i]->labelcolor(FL_RED);
            hs.robot_panel->ws.robot_bat_state[i]->label("0 V"); // battery status
            hs.robot_panel->ws.robot_bat_state[i]->labelcolor(FL_RED);
        }
    }

    // clear message zone
    widgets->msg_zone->label("");
    // unlock config button
    widgets->config->activate();
}

void ToolBar::cb_button_config(Fl_Widget *w, void *data)
{
    
    if (hs.config_dlg != NULL)
    {
        if (hs.config_dlg->shown()) // if shown, do not open again
        {}
        else
        {
            hs.config_dlg->show(); 
        }
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

void ToolBar::cb_button_open(Fl_Widget *w, void *data)
{
    ToolBar_Widgets* widgets = (ToolBar_Widgets*)data;

    if (hs.fc != NULL) {
    }
    else {
        hs.fc = new Fl_File_Chooser(".", "*", Fl_File_Chooser::SINGLE, "Choose data file");
        hs.fc->show();
    }
}

void ToolBar::restore_from_configs(ToolBar_Widgets* ws, void *data)
{
    Config_t* configs = Config_get_configs(); // get runtime configs

    // check whether to open robot panel or not
    if (configs->system.robot_panel_opened) {
        Fl_Window* window = (Fl_Window*)data;
        hs.robot_panel = new RobotPanel(window->x(), window->y()+window->h()+40, 
            window->w(), 200, "Robot Panel");
        hs.robot_panel->robot_button = ws->robot;
        ws->robot->value(1);
    }
    // check whether to open result panel or not
    if (configs->system.result_panel_opened) {
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
    ws.open = new Fl_Button(Xpos, Ypos, Width, Height); Xpos += Width +5;
    ws.msg_zone = new Fl_Box(FL_DOWN_BOX, Xpos, Ypos, bar->w()-Xpos, Height, "");
    ws.msg_zone->align(Fl_Align(FL_ALIGN_CENTER|FL_ALIGN_INSIDE));
    resizable(ws.msg_zone); // protect buttons from resizing
    // icons
    Fl_Pixmap *icon_start = new Fl_Pixmap(pixmap_icon_play);
    Fl_Pixmap *icon_pause = new Fl_Pixmap(pixmap_icon_pause);
    Fl_Pixmap *icon_stop = new Fl_Pixmap(pixmap_icon_stop);
    Fl_Pixmap *icon_config = new Fl_Pixmap(pixmap_icon_config);
    Fl_Pixmap *icon_record = new Fl_Pixmap(pixmap_icon_record);
    Fl_Pixmap *icon_robot = new Fl_Pixmap(pixmap_icon_helicopter);
    Fl_Pixmap *icon_result = new Fl_Pixmap(pixmap_icon_result);
    Fl_Pixmap *icon_open = new Fl_Pixmap(pixmap_icon_open);
    // link icons to buttons
    ws.start->image(icon_start);
    ws.pause->image(icon_pause);
    ws.stop->image(icon_stop);
    ws.config->image(icon_config);
    ws.record->image(icon_record);
    ws.robot->image(icon_robot);
    ws.result->image(icon_result);
    ws.open->image(icon_open);
    // tips for buttons
    ws.start->tooltip("Start Searching");
    ws.pause->tooltip("Pause Searching");
    ws.stop->tooltip("Stop Searching");
    ws.config->tooltip("Settings");
    ws.record->tooltip("Recording");
    ws.robot->tooltip("Robot viewer & controller");
    ws.result->tooltip("Result viewer");
    ws.open->tooltip("Choose data file to replay");
    // types of buttons
    ws.start->type(FL_TOGGLE_BUTTON); // start & pause are mutually exclusive
    ws.pause->type(FL_TOGGLE_BUTTON);
    ws.robot->type(FL_TOGGLE_BUTTON);
    ws.result->type(FL_TOGGLE_BUTTON);
    // colors
    ws.record->selection_color(FL_RED);
    // link call backs to buttons
    ws.start->callback(cb_button_start, (void*)&ws);
    ws.pause->callback(cb_button_pause, (void*)&ws);
    //  start & pause buttons will be released when stop button is pressed
    ws.stop->callback(cb_button_stop, (void*)&ws);
    //  config dialog will pop up when config button pressed
    ws.config->callback(cb_button_config, (void*)win);
    //  robot window will pop up when robot button pressed
    ws.robot->callback(cb_button_robot, (void*)win);
    //  result window will pop up when result button pressed
    ws.result->callback(cb_button_result, (void*)win);
    //  file choosing window will pop up when open button pressed
    ws.open->callback(cb_button_open, (void*)&ws);
    end();
}

/* ====================================
 * ============== UI ==================
 * ==================================== */
void UI::cb_close(Fl_Widget* w, void* data) { 
    // close player
    if (Fl::event() == FL_CLOSE) { 
        //Fl::remove_timeout(cb_repeated_tasks_2hz); // remove timeout callback for repeated tasks
        Fl::remove_timeout(cb_repeated_tasks_10hz); // remove timeout callback for repeated tasks

        UI_Widgets* ws = (UI_Widgets*)data;

        // save open/close states of other sub-panels to configs
        Config_t* configs = Config_get_configs(); // get runtime configs
        if (((ToolBar*)ws->toolbar)->hs.robot_panel != NULL) { // robot panel
            if (((ToolBar*)ws->toolbar)->hs.robot_panel->shown())
                configs->system.robot_panel_opened = true;
            else
                configs->system.robot_panel_opened = false;
        }
        if (((ToolBar*)ws->toolbar)->hs.result_panel != NULL) { // result panel
            if (((ToolBar*)ws->toolbar)->hs.result_panel->shown())
                configs->system.result_panel_opened = true;
            else
                configs->system.result_panel_opened = false;
        }
        // close other panels
        if (((ToolBar*)ws->toolbar)->hs.config_dlg != NULL && ((ToolBar*)ws->toolbar)->hs.config_dlg->shown()) // close config dialog
            ((ToolBar*)ws->toolbar)->hs.config_dlg->hide();
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
    Fl_Double_Window *ui = new Fl_Double_Window(1600, 0, width, height, title);
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
    glutInitWindowPosition(5, tool->h()+5); // place it inside parent window
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
