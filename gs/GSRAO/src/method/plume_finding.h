/*
 * Odor finding
 *
 * Author:
 *      Roice   Luo
 * Date:
 *      2017.06.22
 */

class Plume_Finding
{
    public:
        Plume_Finding(float, float, float, float alpha, float scaler, float vel, float dt);
        void update(void);
        void reinit(float, float, float);
        float current_position[3]; 
    private:
        float type_alpha;
        float cast_scaler;
        float cast_velocity;
        float angle;
        float initial_position[3]; 
        float time_marching_interval;
        int time_marching_idx;
};
