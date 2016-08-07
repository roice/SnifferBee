#ifndef UI_H
#define UI_H

#include "FL/Fl_Widget.H"

struct UI_Widgets {
    void*   toolbar;
};

class UI
{
public:
    UI(int width, int height, const char* title);
    UI_Widgets ws;
private:
    static void cb_close(Fl_Widget *w, void *data);
};

#endif
