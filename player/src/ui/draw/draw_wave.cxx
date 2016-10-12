#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/fl_draw.H>
#include <vector>
#include "ui/draw/draw_wave.h"
#include "io/play_thread.h"
#include "foc/flying_odor_compass.h"

WavePlot::WavePlot(int xpos, int ypos, int width, int height, const char*title=0):Fl_Widget(xpos, ypos, width, height, title)
{
    robot_to_display = 0; // robot 1 by default 
}

void WavePlot::draw(void)
{
    if (!play_thread_get_data())
        return;

    if (robot_to_display >= 4 || robot_to_display < 0)
        robot_to_display = 0; // 0/1/2/3

    std::vector<FOC_Input_t>* data = &(((Flying_Odor_Compass*)play_thread_get_data())->data_raw);

    // draw sensor reading
    const Fl_Color cmap_sensors[3] = {FL_RED, FL_YELLOW, FL_BLUE};
    const float time_range = 10*60; // max display 10 min
    float x1, y1, x2, y2; 
    if (data->size() >= 2) for (int i = 0; i < data->size()-1; i++)
    {
        x1 = x() + (1.0/FOC_MOX_DAQ_FREQ)*(data->at(i).count-data->at(0).count)*w()/time_range;
        x2 = x() + (1.0/FOC_MOX_DAQ_FREQ)*(data->at(i+1).count-data->at(0).count)*w()/time_range;
        for (int idx_sensor = 0; idx_sensor < 3; idx_sensor++)
        {
            y1 = y() + h() - data->at(i).mox_reading[idx_sensor]*h()/3.3; // max 3.3 V
            y2 = y() + h() - data->at(i+1).mox_reading[idx_sensor]*h()/3.3; // max 3.3 V
            fl_color(cmap_sensors[idx_sensor]);
            fl_line(x1, y1, x2, y2);
        }       
    }
}

void WavePlot::Timer_CB(void* data)
{
    ((WavePlot*)data)->redraw();
    Fl::repeat_timeout(0.1, Timer_CB, data);
}

void WavePlot::start(void)
{
    Fl::add_timeout(0.1, Timer_CB, (void*)this);
}

void WavePlot::stop(void)
{
    Fl::remove_timeout(Timer_CB);
}
