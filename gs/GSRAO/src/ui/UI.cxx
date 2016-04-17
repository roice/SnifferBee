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
/* OpenGL */
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <FL/glut.H>
/* RAOS */
#include "ui/UI.h"
#include "ui/icons/icons.h" // pixmap icons used in Tool bar
#include "ui/View.h" // 3D RAO view
#include "GSRAO_Config.h"

/*------- Configuration Dialog -------*/
struct ConfigDlg_widgets { // for parameter saving
    // arena widith/length/height
    Fl_Value_Input *arena_w;
    Fl_Value_Input *arena_l;
    Fl_Value_Input *arena_h;
    // source position xyz
    Fl_Value_Slider *source_x;
    Fl_Value_Slider *source_y;
    Fl_Value_Slider *source_z;
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
    static void cb_change_src_pos_bounds(Fl_Widget*, void*);
    // function to save current value of widgets to runtime configs
    static void save_value_to_configs(ConfigDlg_widgets*);
    // function to get runtime configs to set value of widgets
    static void set_value_from_configs(ConfigDlg_widgets*);
};

void ConfigDlg::cb_dlg(Fl_Widget* w, void* data) {
    if (Fl::event() == FL_CLOSE) {
        struct ConfigDlg_widgets *ws = (struct ConfigDlg_widgets*)data;
        // save widget values to RAOS runtime configs when closing the dialog window
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

// change source position bounds according to arena size
void ConfigDlg::cb_change_src_pos_bounds(Fl_Widget* arena_wlh, void* src_xyz) {
    ((Fl_Value_Slider*)src_xyz)->minimum(-((Fl_Valuator*)arena_wlh)->value()/2.0);
    ((Fl_Value_Slider*)src_xyz)->maximum(((Fl_Valuator*)arena_wlh)->value()/2.0);
}

void ConfigDlg::save_value_to_configs(ConfigDlg_widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
    // save arena size
    configs->arena.w = ws->arena_w->value();
    configs->arena.l = ws->arena_l->value();
    configs->arena.h = ws->arena_h->value();
    // save source pos
    configs->source.x = ws->source_x->value();
    configs->source.y = ws->source_y->value();
    configs->source.z = ws->source_z->value();
}

void ConfigDlg::set_value_from_configs(ConfigDlg_widgets* ws) {
    GSRAO_Config_t* configs = GSRAO_Config_get_configs(); // get runtime configs
    // set arena size
    ws->arena_w->value(configs->arena.w);
    ws->arena_l->value(configs->arena.l);
    ws->arena_h->value(configs->arena.h);
    // set source pos & maximum
    ws->source_x->value(configs->source.x);
    ws->source_y->value(configs->source.y);
    ws->source_z->value(configs->source.z);
    ws->source_x->minimum(-configs->arena.w/2.0);
    ws->source_y->minimum(-configs->arena.l/2.0);
    ws->source_z->minimum(-configs->arena.h/2.0);
    ws->source_x->maximum(configs->arena.w/2.0);
    ws->source_y->maximum(configs->arena.l/2.0);
    ws->source_z->maximum(configs->arena.h/2.0);
}

ConfigDlg::ConfigDlg(int xpos, int ypos, int width, int height, 
        const char* title=0):Fl_Window(xpos,ypos,width,height,title)
{
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
            // Arena
            Fl_Box *arena = new Fl_Box(t_x+10, t_y+25+10, 180, 130,"Arena");
            arena->box(FL_PLASTIC_UP_FRAME);
            arena->labelsize(16);
            arena->labelfont(FL_COURIER_BOLD_ITALIC);
            arena->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            // Arena width/length/height
            dlg_w.arena_w = new Fl_Value_Input(t_x+10+60,t_y+25+10+30,80,25,"Width");
            dlg_w.arena_l = new Fl_Value_Input(t_x+10+60,t_y+25+10+60,80,25,"Length");
            dlg_w.arena_h = new Fl_Value_Input(t_x+10+60,t_y+25+10+90,80,25,"Height");
            new Fl_Box(t_x+10+60+80,t_y+25+10+30, 20, 25, "m");
            new Fl_Box(t_x+10+60+80,t_y+25+10+60, 20, 25, "m");
            new Fl_Box(t_x+10+60+80,t_y+25+10+90, 20, 25, "m");
            // Source
            Fl_Box *source = new Fl_Box(t_x+10+190, t_y+25+10, 180, 130,"Source Position");
            source->box(FL_PLASTIC_UP_FRAME);
            source->labelsize(16);
            source->labelfont(FL_COURIER_BOLD_ITALIC);
            source->align(Fl_Align(FL_ALIGN_TOP|FL_ALIGN_INSIDE));
            // Source pos
            dlg_w.source_x = new Fl_Value_Slider(t_x+10+210,t_y+25+10+30,140,25,"X");
            dlg_w.source_y = new Fl_Value_Slider(t_x+10+210,t_y+25+10+60,140,25,"Y");
            dlg_w.source_z = new Fl_Value_Slider(t_x+10+210,t_y+25+10+90,140,25,"Z");
            dlg_w.source_x->labelsize(16);
            dlg_w.source_y->labelsize(16);
            dlg_w.source_z->labelsize(16);
            dlg_w.source_x->type(FL_HOR_NICE_SLIDER);
            dlg_w.source_y->type(FL_HOR_NICE_SLIDER);
            dlg_w.source_z->type(FL_HOR_NICE_SLIDER);
            dlg_w.source_x->align(Fl_Align(FL_ALIGN_LEFT));
            dlg_w.source_y->align(Fl_Align(FL_ALIGN_LEFT));
            dlg_w.source_z->align(Fl_Align(FL_ALIGN_LEFT));
            new Fl_Box(t_x+10+210+140,t_y+25+10+30, 20, 25, "m");
            new Fl_Box(t_x+10+210+140,t_y+25+10+60, 20, 25, "m");
            new Fl_Box(t_x+10+210+140,t_y+25+10+90, 20, 25, "m");
            dlg_w.arena_w->callback(cb_change_src_pos_bounds, (void*)dlg_w.source_x);
            dlg_w.arena_l->callback(cb_change_src_pos_bounds, (void*)dlg_w.source_y);
            dlg_w.arena_h->callback(cb_change_src_pos_bounds, (void*)dlg_w.source_z);
        }
        scenario->end();
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
